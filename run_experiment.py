from src.args import argument_parser
from src.dataset import read_dataset
from src.pso import TrackerSwarm, get_roi

import torch
import cv2


def main():
    args = argument_parser().parse_args()

    frame, (x, y, target_width, target_height) = iter(read_dataset(args.data_dir)).__next__()
    frame_height, frame_width, _ = frame.shape

    target_pos = torch.tensor([[x + target_width // 2, y + target_height // 2]], dtype=torch.int)
    target_roi = get_roi(frame, target_pos, target_width, target_height)

    search_space_bounds = {
        'left': target_width // 2,
        'top': target_height // 2,
        'right': frame_width - target_width // 2,
        'bottom': frame_height - target_height // 2
    }

    tracker = TrackerSwarm(
        num_particles=args.num_particles,
        search_space_bounds=search_space_bounds,
        target_pos=target_pos
    )

    last_frame = frame.copy()

    for frame, (x1, y1, tw, th) in read_dataset(args.data_dir):
        for _ in range(args.num_iterations):
            pt1, pt2 = tracker.training_step(frame, last_frame, target_roi,
                                             args.w, args.c1, args.c2,
                                             target_width, target_height)
        last_frame = frame.copy()
        image = cv2.rectangle(frame, pt1, pt2, (255,0,0), 2)
        image = cv2.rectangle(image, (x1, y1), (x1+tw, y1+th), (0,0,255), 2)
        cv2.putText(image, 'Ground truth', (0, 50), 0, 1, (0,0,255), 2)
        cv2.putText(image, 'PSO', (0, 100), 0, 1, (255,0,0), 2)
        cv2.imshow('TrackerSwarm', image)

        tracker.reset_swarm()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()