import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel


class SimpleMLP3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


class HuggingFaceTimeSeriesModel(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        # Configure transformer for time series forecasting
        config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=seq_len,
            input_size=input_size,
            num_time_features=input_size,
            num_static_categorical_features=0,
            num_static_real_features=0,
        )
        self.model = TimeSeriesTransformerModel(config)
        # Final linear layer to produce buy/hold/sell logits
        self.fc = nn.Linear(config.d_model, 3)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape

        # Required: dummy time features (all zeros for now)
        past_time_features = torch.zeros(
            batch_size, seq_len, input_size, device=x.device, dtype=x.dtype
        )

        # Required: observed mask (all ones means all values are valid)
        past_observed_mask = torch.ones(
            batch_size, seq_len, device=x.device, dtype=x.dtype
        )

        # Call the transformer model with all required arguments
        output = self.model(
            past_values=x,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
        ).last_hidden_state
