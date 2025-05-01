import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

def get_results():
    try:
        header = sys.stdin.readline().strip().split()
        if len(header) != 2:
            raise ValueError("Invalid header format")
        num_bodies, seconds = int(header[0]), int(header[1])
    except (ValueError, IndexError) as e:
        print(f"Error reading header: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    radii = []
    
    try:
        for _ in range(seconds):
            frame = []
            for _ in range(num_bodies):
                line = sys.stdin.readline().strip()
                if not line:
                    raise ValueError("Unexpected end of input")
                x, y, z, radius = map(float, line.split())
                frame.append([x, y, z])
                if len(radii) < num_bodies:
                    radii.append(radius)
            results.append(np.array(frame))
    except Exception as e:
        print(f"Error reading data: {e}", file=sys.stderr)
        sys.exit(1)

    return results, radii

def update(iteration, results, scat):
    current_pos = results[iteration]
    scat._offsets3d = (current_pos[:, 0], current_pos[:, 1], current_pos[:, 2])
    return scat,

def animate(results, radii):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    results = np.array(results)
    x = results[0, :, 0]
    y = results[0, :, 1]
    z = results[0, :, 2]

    max_r = max(radii)
    sizes = [(r / max_r) * 100 for r in radii]
    print(len(x), len(y), len(z), len(sizes))
    
    scat = ax.scatter(
        x, y, z,
        s=sizes,
        c='cyan',
        marker='o',
        alpha=1,
        depthshade=False
    )
    
    ax.set_xlim(abs(min(x)) * -2, max(x) * 2)
    ax.set_ylim(abs(min(y)) * -2, max(y) * 2)
    ax.set_zlim(abs(min(z)) * -2, max(z) * 2)
    ax.set_facecolor('black')
    ax.grid(False)
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(results),
        fargs=(results, scat),
        interval=50, blit=False, repeat=True
    )
    ani.save('nbody.gif', writer='pillow', fps=15, dpi=100)
    plt.show()    
    

if __name__ == '__main__':
    results, radii = get_results()
    print(len(results))
    animate(results, radii)
