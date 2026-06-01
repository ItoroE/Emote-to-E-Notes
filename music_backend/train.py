import torch
import torch.nn as nn
import torch.nn.functional as F

class EmoCVAE(nn.Module):
    def __init__(
        self,
        note_dim=2048,
        hidden_dim=256,
        z_dim=16,
        condition_dim=2,
        num_notes = 0,
        beta = 0.001,
        lambda_emo=0.1
    ):
        super().__init__()

        self.numnotes = num_notes # is just comprised of num all note_events 
        self.note_dim = note_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.condition_dim = condition_dim
        self.beta = beta
        self.lambda_emo = lambda_emo
        self.relu = nn.ReLU()

        # encoder
        self.encoder_gru = nn.GRU(note_dim + condition_dim, 
                                  hidden_size = hidden_dim,
                                  batch_first = True) # specification on the order in which I wrote batch size versus hidden dim size
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

        # decoder
        self.decoder_input = nn.Linear(z_dim + condition_dim, hidden_dim)
        self.decoder_gru = nn.GRU(
                input_size =hidden_dim,
                hidden_size = hidden_dim,
                batch_first= True
                )
        self.output = nn.Linear(hidden_dim, note_dim)

        # emotion prediction head, saves collapse
        self.emotion_head = nn.Linear(z_dim, condition_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c):
       #needed for nn.Linear x = x.view(x.size(0), -1) 
        # goes from [batch, cdim] to [batch, numnotes, cdim]

        #print("x:", x.shape)
        #print("c:", c.shape)
        c_unzipped = c.unsqueeze(1).expand(-1, x.size(1), -1)

        #print("c_unzipped:",c_unzipped.shape)
        
        xc = torch.cat([x, c_unzipped], dim=2)
        output, h_n = self.encoder_gru(xc)
        h = h_n[-1]
       
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        return mu, logvar

    def decode(self, z, c, numnotes):
        zc = torch.cat([z, c], dim=1)
        h = self.relu(self.decoder_input(zc))
        sequence= h.unsqueeze(1).expand(-1, numnotes, -1)
        dec_output, _ = self.decoder_gru(sequence)
        recon = self.output(dec_output)

        # apply sigmoid to pitch and velocity  to avoid negative nums
        recon[:,:,0] = torch.sigmoid(recon[:,:,0])  # pitch
        recon[:,:,3] = torch.sigmoid(recon[:,:,3])  # velocity
        recon[:,:,1] = torch.sigmoid(recon[:,:,1])  # start
        recon[:,:,2] = torch.sigmoid(recon[:,:,2])  # duration
        return recon

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        emo_pred = self.emotion_head(z)
        return recon, mu, logvar, emo_pred