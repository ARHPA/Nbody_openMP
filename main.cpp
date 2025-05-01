#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <getopt.h>
#include <fstream>
#include <string>
#include <ctime>
#include <limits>
#include <queue>
#include <sstream>
#include <chrono>

using namespace std;

struct Vector3D {
    double x, y, z;
    Vector3D operator+(const Vector3D& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    Vector3D& operator+=(const Vector3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    Vector3D operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
};

struct alignas(128) Body {
    Vector3D pos;
    Vector3D vel; 
    Vector3D acc; 
    double mass; 
    double radius; 
    bool collided;  
    
    char padding[128 - (sizeof(pos) + sizeof(vel) + sizeof(acc) + 
                       sizeof(mass) + sizeof(radius) + sizeof(collided))];
};

class OctreeNode {
public:
    Vector3D minBound, maxBound;
    Vector3D centerOfMass;
    double mass;
    vector<OctreeNode*> children;
    const Body* body = nullptr;
    
    OctreeNode(const Vector3D& min, const Vector3D& max) 
        : minBound(min), maxBound(max), mass(0) {}
        
    bool isLeaf() const { return children.empty(); }
};

class BarnesHutTree {
    double theta;
    OctreeNode* root;
    
public:
    BarnesHutTree(const vector<Body>& bodies, double theta = 0.5) {
        this->theta = theta;

        double minX = numeric_limits<double>::max();
        double minY = numeric_limits<double>::max();
        double minZ = numeric_limits<double>::max();
        double maxX = -numeric_limits<double>::max();
        double maxY = -numeric_limits<double>::max();
        double maxZ = -numeric_limits<double>::max();

        #pragma omp parallel for reduction(min:minX, minY, minZ) reduction(max:maxX, maxY, maxZ)
        for (int i = 0; i < bodies.size(); ++i) {
            const Vector3D& p = bodies[i].pos;
            minX = min(minX, p.x);
            minY = min(minY, p.y);
            minZ = min(minZ, p.z);
            maxX = max(maxX, p.x);
            maxY = max(maxY, p.y);
            maxZ = max(maxZ, p.z);
        }

        Vector3D minBound = {minX, minY, minZ};
        Vector3D maxBound = {maxX, maxY, maxZ};

        root = new OctreeNode(minBound, maxBound);
        
        #pragma omp parallel for
        for (int i = 0; i < bodies.size(); ++i) {
            #pragma omp critical
            insertBody(root, &bodies[i]);
        }
        
        #pragma omp parallel
        {
            #pragma omp single
            computeMassDistribution(root);
        }
    }

    void insertBody(OctreeNode* node, const Body* body) {
        if (node->isLeaf()) {
            if (node->body == nullptr) {
                node->body = body;
            } else {
                subdivide(node);
                insertBody(node, body);
                insertBody(node, node->body);
                node->body = nullptr;
            }
        } else {
            int octant = getOctant(node, body->pos);
            insertBody(node->children[octant], body);
        }
    }

    void computeMassDistribution(OctreeNode* node) {
        if (node->isLeaf()) {
            if (node->body) {
                node->mass = node->body->mass;
                node->centerOfMass = node->body->pos;
            }
            return;
        }

        node->mass = 0;
        Vector3D com{0,0,0};

        #pragma omp taskgroup
        {
            for (int i = 0; i < 8; ++i) {
                #pragma omp task firstprivate(i) shared(node)
                computeMassDistribution(node->children[i]);
            }
        }

        for (int i = 0; i < 8; ++i) {
            auto* child = node->children[i];
            node->mass += child->mass;
            com.x += child->centerOfMass.x * child->mass;
            com.y += child->centerOfMass.y * child->mass;
            com.z += child->centerOfMass.z * child->mass;
        }

        node->centerOfMass.x = com.x / node->mass;
        node->centerOfMass.y = com.y / node->mass;
        node->centerOfMass.z = com.z / node->mass;
    }

    Vector3D computeForce(const Body* body, double G, double softening) const {
        Vector3D totalForce = {0, 0, 0};
        vector<OctreeNode*> currentLevel;
        currentLevel.push_back(root);

        while (!currentLevel.empty()) {
            vector<OctreeNode*> nextLevel;
            Vector3D levelForce = {0, 0, 0};

            #pragma omp parallel
            {
                Vector3D localForce = {0, 0, 0};
                vector<OctreeNode*> localNextLevel;

                #pragma omp for schedule(dynamic)
                for (int i = 0; i < currentLevel.size(); ++i) {
                    OctreeNode* node = currentLevel[i];
                    if (node == nullptr) continue;

                    const double dx = node->centerOfMass.x - body->pos.x;
                    const double dy = node->centerOfMass.y - body->pos.y;
                    const double dz = node->centerOfMass.z - body->pos.z;
                    const double distSq = dx*dx + dy*dy + dz*dz + softening*softening;

                    if (node->isLeaf() && node->body == body) continue;

                    const double s = max(max(node->maxBound.x - node->minBound.x,
                                        node->maxBound.y - node->minBound.y),
                                        node->maxBound.z - node->minBound.z);
                    const double d = sqrt(distSq);

                    if (node->isLeaf() || (s/d < theta)) {
                        if (node->mass == 0) continue;

                        const double invDist = 1.0 / sqrt(distSq);
                        const double invDist3 = invDist * invDist * invDist;
                        const double F = G * body->mass * node->mass * invDist3;

                        localForce.x += F * dx;
                        localForce.y += F * dy;
                        localForce.z += F * dz;
                    } else {
                        for (auto child : node->children) {
                            if (child != nullptr) {
                                localNextLevel.push_back(child);
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    levelForce.x += localForce.x;
                    levelForce.y += localForce.y;
                    levelForce.z += localForce.z;
                }

                #pragma omp critical
                {
                    nextLevel.insert(nextLevel.end(), localNextLevel.begin(), localNextLevel.end());
                }
            }

            totalForce += levelForce;
            currentLevel = move(nextLevel);
        }

        return totalForce;
    }

private:
    void subdivide(OctreeNode* node) {
        const Vector3D center = {
            (node->minBound.x + node->maxBound.x) / 2,
            (node->minBound.y + node->maxBound.y) / 2,
            (node->minBound.z + node->maxBound.z) / 2
        };

        for (int i = 0; i < 8; ++i) {
            Vector3D newMin, newMax;

            if (i & 1) {
                newMin.x = center.x;
                newMax.x = node->maxBound.x;
            } else {
                newMin.x = node->minBound.x;
                newMax.x = center.x;
            }

            if (i & 2) {
                newMin.y = center.y;
                newMax.y = node->maxBound.y;
            } else {
                newMin.y = node->minBound.y;
                newMax.y = center.y;
            }

            if (i & 4) {
                newMin.z = center.z;
                newMax.z = node->maxBound.z;
            } else {
                newMin.z = node->minBound.z;
                newMax.z = center.z;
            }

            node->children.push_back(new OctreeNode(newMin, newMax));
        }
    }

    int getOctant(const OctreeNode* node, const Vector3D& pos) const {
        const Vector3D center = {
            (node->minBound.x + node->maxBound.x) / 2,
            (node->minBound.y + node->maxBound.y) / 2,
            (node->minBound.z + node->maxBound.z) / 2
        };
        
        bool rightOfCenter  = (pos.x > center.x);
        bool aboveCenter    = (pos.y > center.y);
        bool frontOfCenter  = (pos.z > center.z);

        int octant = 0;
        if (rightOfCenter) octant += 1;
        if (aboveCenter)   octant += 2;
        if (frontOfCenter) octant += 4;
        return octant;
    }
};

class NBodySolver {
    const double G = 6.67430e-11;
    const double softening = 10;
    double timeStep = 1.0;
    double theta = 0.7;

public:
    void simulate(vector<Body>& bodies, int seconds, int num_threads, bool output) {
        omp_set_num_threads(num_threads);
        const int N = bodies.size();

        if (output) {
            cout << N << " " << seconds << endl;
            cout.flush();
        }

        double total_time = 0;
        // auto simulation_start = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < seconds; ++s) {
            auto iteration_start = std::chrono::high_resolution_clock::now();
            BarnesHutTree tree(bodies, theta);

            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    Vector3D force = tree.computeForce(&bodies[i], G, softening);
                    bodies[i].acc = {force.x / bodies[i].mass, 
                                    force.y / bodies[i].mass, 
                                    force.z / bodies[i].mass};
                }

                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    bodies[i].vel += bodies[i].acc * timeStep;
                    bodies[i].pos += bodies[i].vel * timeStep;
                }
            }

            auto iteration_end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double>(iteration_end - iteration_start).count();

            if (output) {
                for (int i = 0; i < N; ++i) {
                    cout << bodies[i].pos.x << " " 
                         << bodies[i].pos.y << " " 
                         << bodies[i].pos.z << " "
                         << bodies[i].radius << "\n";
                }
                cout.flush();
            }
        }
        auto simulation_end = std::chrono::high_resolution_clock::now();
        // double total_wall_time = std::chrono::duration<double>(simulation_end - simulation_start).count();
        
        std::cerr << "\nSimulation time (computation only): " << total_time << " sec\n";
        // std::cerr << "Total wall-clock time: " << total_wall_time << " sec\n";
    }

};

int main(int argc, char* argv[]) {
    bool output = false;
    int num_bodies = 3, seconds = 10, num_threads = 4;
    string filename;
    bool use_file = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:t:s:f:o")) != -1) {
        switch (opt) {
            case 'o': output = true; break;
            case 'n': num_bodies = atoi(optarg); break;
            case 't': num_threads = atoi(optarg); break;
            case 's': seconds = atoi(optarg); break;
            case 'f': filename = optarg; use_file = true; break;
            default: 
                cerr << "Usage: " << argv[0] << " [-n bodies] [-t threads] [-s seconds] [-f input_file]\n";
                return 1;
        }
    }

    vector<Body> bodies;
    
    if (use_file) {
        ifstream file(filename);
        if (!file) {
            cerr << "Error opening file: " << filename << endl;
            return 1;
        }
        int n;
        file >> n;
        for (int i = 0; i < n; ++i) {
            double x, y, z, mass, radius;
            file >> x >> y >> z >> mass >> radius;
            bodies.push_back({{x, y, z}, {0, 0, 0}, {0, 0, 0}, mass, radius, false});
        }
    } else {
        srand(time(NULL));
        for (int i = 0; i < num_bodies; ++i) {
            Vector3D pos = {
                static_cast<double>(rand()%5000 - 2500),
                static_cast<double>(rand()%5000 - 2500),
                static_cast<double>(rand()%5000 - 2500)
            };
            Vector3D vel = {0.0, 0.0, 0.0};
            Vector3D acc = {0.0, 0.0, 0.0};
            double mass = static_cast<double>(rand()%1000 + 1) * 1e13;
            double radius = static_cast<double>(rand()%100 + 1);
            
            bodies.push_back(Body{pos, vel, acc, mass, radius, false});
        }
    }

    NBodySolver solver;
    solver.simulate(bodies, seconds, num_threads, output);
    cout << bodies.size() << " " << seconds << endl;
    for (auto& b : bodies) {
        cout << b.pos.x << " " << b.pos.y << " " << b.pos.z << " " << b.radius << endl;
    }

    return 0;
}