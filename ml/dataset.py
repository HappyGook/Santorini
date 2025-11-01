import numpy as np

from ml.encode import encode_board, encode_action


class SantoDataset:
    def __init__(self):
        self.states = []
        self.actions = []
        self.scores=[]

    def add_sample(self, board, player_id, action, score):
        state_tensor = encode_board(board, player_id)
        action_tensor = encode_action(*action)
        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        score = min(1, max(-1, score/10))
        self.scores.append(score)

    def save(self, path):
        if len(self.states) == 0:
            # Avoiding empty stack with placeholders
            np.savez_compressed(
                path,
                states=np.zeros((0, 6, 5, 5), dtype=np.float32),
                actions=np.zeros((0, 3, 5, 5), dtype=np.float32),
                values=np.zeros((0,), dtype=np.float32),
            )
            print(f"[DATASET] Saved empty dataset to {path}")
            return

        np.savez_compressed(
            path,
            states=np.stack(self.states),
            actions=np.stack(self.actions),
            values=np.array(self.scores, dtype=np.float32),
        )
        print(f"[DATASET] Saved {len(self.states)} samples to {path}")

    @staticmethod
    def load(path):
        data = np.load(path)
        ds = SantoDataset()
        ds.states = list(data["states"])
        ds.actions = list(data["actions"])
        ds.scores = list(data["scores"])
        return ds


if __name__ == "__main__":
    ds = SantoDataset()
    ds.save("test.npz")