import numpy as np
import torch


def get_roi(frame, centroid, width, height, only_points=False):
    points = torch.cat((
        centroid[:,0,None] - width // 2,
        centroid[:,0,None] + width // 2,
        centroid[:,1,None] - height // 2,
        centroid[:,1,None] + height // 2,
    ), dim=1)

    if only_points:
        points = points.numpy()
        return (points[0][0], points[0][2]), (points[0][1], points[0][3])

    return torch.from_numpy(
        np.array(
            [frame[y0:y1, x0:x1] for x0, x1, y0, y1 in points]
        ).astype('float32') / 0xFF
    )


def get_bounds(position, search_space_bounds, search_size=30):
    search_size = torch.tensor([-search_size, search_size], dtype=torch.int)
    xlim = torch.clip(
        position[0][0] + search_size,
        min=search_space_bounds['left'],
        max=search_space_bounds['right']
    )
    ylim = torch.clip(
        position[0][1] + search_size,
        min=search_space_bounds['top'],
        max=search_space_bounds['bottom']
    )
    return xlim.numpy(), ylim.numpy()


class TrackerSwarm:
    def __init__(
        self,
        num_particles: int,
        search_space_bounds,
        target_pos,
    ):
        self.num_particles = num_particles
        self.target_pos = target_pos.clone()
        self.gbest_pos = target_pos.clone()
        self.search_space_bounds = search_space_bounds
        self.reset_swarm()

    def relocate(self):
        xlim, ylim = get_bounds(self.gbest_last_pos, self.search_space_bounds)
        new_pos = torch.cat((
            torch.randint(xlim[0], xlim[1] + 1, (self.num_particles, 1)).int(),
            torch.randint(ylim[0], ylim[1] + 1, (self.num_particles, 1)).int(),
        ), dim=1)
        return new_pos

    def update_velocity(self, w, c1, c2):
        self.velocity = w * self.velocity + \
            c1 * torch.rand(self.num_particles, 2) * (self.pbest_pos - self.curr_pos) + \
            c2 * torch.rand(self.num_particles, 2) * (self.gbest_pos - self.curr_pos)

    def update_position(self):
        xlim, ylim = get_bounds(self.gbest_last_pos, self.search_space_bounds)
        self.curr_pos = torch.round(self.curr_pos + self.velocity).int()
        self.curr_pos[:, 0] = torch.clip(self.curr_pos[:, 0], xlim[0], xlim[1])
        self.curr_pos[:, 1] = torch.clip(self.curr_pos[:, 1], ylim[0], ylim[1])

    def evaluate(self, curr_frame, last_frame, target_roi):
        _, height, width, _ = target_roi.size()
        curr_roi = get_roi(curr_frame, self.curr_pos, width, height)
        last_roi = get_roi(last_frame, self.gbest_last_pos, width, height)
        scores = torch.norm((curr_roi - last_roi).view(self.num_particles, -1), dim=1)
        scores += torch.norm((curr_roi - target_roi).view(self.num_particles, -1), dim=1)

        indices = scores < self.pbest_score
        self.pbest_score[indices] = scores[indices]
        self.pbest_pos[indices] = self.curr_pos[indices]
        min_score, indice = torch.min(scores, dim=0)
        if min_score < self.gbest_score:
            self.gbest_score = min_score
            self.gbest_pos = self.curr_pos[indice, None]

    def training_step(
        self,
        frame,
        last_frame,
        target_roi,
        w,
        c1,
        c2,
        target_width,
        target_height,
    ):
        self.evaluate(frame, last_frame, target_roi)
        self.update_velocity(w, c1, c2)
        self.update_position()
        pt1, pt2 = get_roi(
            frame,
            self.gbest_pos,
            target_width,
            target_height,
            only_points=True,
        )
        return (pt1, pt2)

    def reset_swarm(self):
        self.gbest_last_pos = self.gbest_pos.clone()
        self.gbest_score = float('inf')
        self.velocity = torch.zeros((self.num_particles, 2))
        self.curr_pos = self.relocate()
        self.pbest_pos = self.curr_pos.clone()
        self.pbest_score = torch.full((self.num_particles,), float('inf'))