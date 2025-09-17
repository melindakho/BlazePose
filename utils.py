import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mediapipe as mp

LEFT_LANDMARKS = [
    4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
]

RIGHT_LANDMARKS = [
    1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
]

def animate(poses_df, save_path=None, fps=33):
    """Animate landmarks from a DataFrame loaded from CSV."""
    mp_pose = mp.solutions.pose
    connections = list(mp_pose.POSE_CONNECTIONS)
    n_landmarks = len([col for col in poses_df.columns if col.startswith("x_")])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invert y for image-like coordinates

    scat = ax.scatter([], [], s=30, c='r')
    lines = []
    for start_idx, end_idx in connections:
        if start_idx in LEFT_LANDMARKS and end_idx in LEFT_LANDMARKS:
            color = 'b'
        elif start_idx in RIGHT_LANDMARKS and end_idx in RIGHT_LANDMARKS:
            color = 'r'
        else:
            color = 'y'
        line, = ax.plot([], [], lw=2, c=color)
        lines.append(line)

    def update(frame_idx):
        row = poses_df.iloc[frame_idx]
        xs = [row[f"x_{i}"] for i in range(n_landmarks)]
        ys = [row[f"y_{i}"] for i in range(n_landmarks)]
        colors = []
        for i in range(n_landmarks):
            if i in LEFT_LANDMARKS:
                colors.append('b')
            elif i in RIGHT_LANDMARKS:
                colors.append('r')
            else:
                colors.append('g')
        scat.set_offsets(list(zip(xs, ys)))
        scat.set_color(colors)
        for i, (start, end) in enumerate(connections):
            lines[i].set_data([xs[start], xs[end]], [ys[start], ys[end]])
        return [scat] + lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(poses_df), interval=30, blit=True, repeat=True
    )

    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)
    else:
        plt.show()