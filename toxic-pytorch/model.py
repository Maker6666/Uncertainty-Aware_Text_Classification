import torch
import torch.nn as nn
from scipy.stats import hmean
import torch.nn.functional as F


class TextCnn(nn.Module):
    def __init__(self, embed_dim, kernel_num, kernel_sizes, dropout=0.5):
        super(TextCnn, self).__init__()

        in_channels = 1
        out_channels = kernel_num

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.mu_backbone = nn.Sequential(
            nn.Linear(kernel_num * len(kernel_sizes), embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.logvar_backbone = nn.Sequential(
            nn.Linear(kernel_num * len(kernel_sizes), embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, embedded):
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in
                  self.convs]
        pooled = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in
                  conved]
        x = torch.cat(pooled, 1)
        x = self.dropout(x)

        mu = self.mu_backbone(x)
        logvar = self.logvar_backbone(x)
        std = (logvar * 0.5).exp()
        return mu, std


class MultiLabelClassification(nn.Module):
    def __init__(self, textcnn: TextCnn, input_dim: int, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

        self.textcnn = textcnn
        self.classifier = nn.Linear(input_dim, num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(
            self,
            inputs,
            labels=None,
            label_idx=None,
            istrain=False
    ):

        mu, std = self.textcnn(inputs)
        if istrain:
            epsilon = torch.randn_like(std)
            features = mu + epsilon * std
        else:
            features = mu
        features = torch.cat((features, labels[:, :label_idx]), dim=-1)
        variance = std ** 2
        loss_kl = ((variance + mu ** 2 - torch.log(variance) - 1) * 0.5).sum(dim=-1).mean()
        loss_std = torch.tensor(hmean(std.cpu().detach().numpy(), axis=1).mean())

        outputs = self.classifier(features)

        # add hidden states and attention if they are here
        loss = 0
        if labels is not None:
            y_true = labels[:, label_idx].to(torch.float).unsqueeze(1)
            loss += 0.0001 * loss_kl + 0.1 * loss_std
            loss += self.loss_fct(outputs, y_true)
        return loss, outputs

