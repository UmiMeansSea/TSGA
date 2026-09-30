import torch
import torch.nn as nn
import torch.nn.functional as F

class TSPDCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TSPDCNN, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 1D Convolution to extract features from the distance matrix rows
        # Treating each city's distances to other cities as input channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=embedding_dim, kernel_size=3, padding=1)
        
        # Decoder parameters to score next-city probabilities
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, distance_matrix, return_pi=False):
        """
        distance_matrix: Tensor of shape (batch_size, n, n)
        Returns: tour (batch_size, n), log_probs (batch_size)
        """
        batch_size, n, _ = distance_matrix.shape
        device = distance_matrix.device

        # Embed cities via CNN
        # Reshape for Conv1D: (batch_size * n, 1, n)
        x = distance_matrix.view(batch_size * n, 1, n).float()
        
        # Convolutions to create node embeddings
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) # shape: (batch_size * n, embedding_dim, n)
        
        # Global average pooling across the n dimension to get 1 embedding per city
        node_embeddings = x.mean(dim=2).view(batch_size, n, self.embedding_dim) 

        # RL Decoding (Constructive Phase)
        tours = []
        log_probs = torch.zeros(batch_size, device=device)
        
        # Boolean mask to track visited cities (True = visited)
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        
        # Start at city 0 for simplicity (TSP is a closed cycle, starting point doesn't matter)
        current_city = torch.zeros(batch_size, dtype=torch.long, device=device)
        tours.append(current_city)
        mask[torch.arange(batch_size), current_city] = True

        for step in range(1, n):
            # Query based on current city embedding
            query = node_embeddings[torch.arange(batch_size), current_city]
            query = self.W_q(query).unsqueeze(1) # (batch, 1, dim)
            
            # Keys based on all city embeddings
            keys = self.W_k(node_embeddings) # (batch, n, dim)
            
            # Attention scoring
            scores = torch.sum(self.v * torch.tanh(query + keys), dim=2) # (batch, n)
            
            # Mask out already visited cities by setting their score to -infinity
            scores = scores.masked_fill(mask, float('-inf'))
            
            # Convert scores to probabilities
            probs = F.softmax(scores, dim=1)
            
            # Sample next city during training, or take argmax during evaluation
            if self.training:
                m = torch.distributions.Categorical(probs)
                next_city = m.sample()
                log_probs += m.log_prob(next_city)
            else:
                next_city = probs.argmax(dim=1)
                
            tours.append(next_city)
            mask = mask.clone()
            mask[torch.arange(batch_size), next_city] = True
            current_city = next_city

        # Stack into shape (batch_size, n)
        tour_tensor = torch.stack(tours, dim=1)
        
        if return_pi:
            return tour_tensor, log_probs
        return tour_tensor