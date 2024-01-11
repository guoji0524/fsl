import torch
import random

class FewShotTrainer:
    def __init__(self):
        pass

    def random_sample(self, data, num_samples):
        return random.sample(data, num_samples)

    def compute_distance(self, embedding, prototype):
        distance = torch.norm(embedding - prototype)
        return distance

    def compute_prototype(self, support_examples):
        prototype = torch.mean(support_examples, dim=0)
        return prototype

    def compute_loss(self, query_example, prototype, num_classes):
        loss = 0.0
        for k0 in range(1, num_classes+1):
            distance = self.compute_distance(query_example, prototype[k0])
            loss += torch.exp(-distance)
        loss = -torch.log(loss / num_classes)
        return loss

    def train_episode(self, training_set, num_classes, num_support, num_query):
        episode_loss = 0.0
        class_indices = self.random_sample(range(1, num_classes+1), num_classes)

        for k in class_indices:
            Dk = [(x, y) for (x, y) in training_set if y == k]
            Sk = self.random_sample(Dk, num_support)
            Qk = self.random_sample([d for d in Dk if d not in Sk], num_query)

            prototype = self.compute_prototype(torch.stack([x for (x, _) in Sk]))

            for (x, y) in Qk:
                x = x.unsqueeze(0)
                loss = self.compute_loss(x, prototype, num_classes)
                episode_loss += loss.item()

        episode_loss /= (num_classes * num_query)
        return episode_loss

trainer = FewShotTrainer()

