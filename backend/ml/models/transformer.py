import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class JazzTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for jazz solo generation.
    
    Encoder: processes chord progressions with metadata
    Decoder: generates melody tokens autoregressively
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_encoder_len=700,
        max_decoder_len=4000,
        pad_id=0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_id = pad_id
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        # Positional encoding
        self.encoder_pos = PositionalEncoding(d_model, max_encoder_len, dropout)
        self.decoder_pos = PositionalEncoding(d_model, max_decoder_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        # src: (batch, src_len)
        # Returns: (batch, src_len) boolean mask where True = ignore
        return src == self.pad_id
    
    def make_tgt_mask(self, tgt):
        # tgt: (batch, tgt_len)
        # Returns causal mask and padding mask
        tgt_len = tgt.size(1)
        
        # Causal mask: (tgt_len, tgt_len)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Padding mask: (batch, tgt_len)
        padding_mask = tgt == self.pad_id
        
        return causal_mask, padding_mask
    
    def encode(self, src, src_key_padding_mask=None):
        """Encode source sequence (chords)."""
        # src: (batch, src_len)
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.encoder_pos(src_emb)
        
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask
        )
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Decode target sequence (melody)."""
        # tgt: (batch, tgt_len)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pos(tgt_emb)
        
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return output
    
    def forward(self, src, tgt):
        """
        Forward pass for training.
        
        Args:
            src: (batch, src_len) - encoder input (chords)
            tgt: (batch, tgt_len) - decoder input (melody, shifted right)
        
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Create masks
        src_key_padding_mask = self.make_src_mask(src)
        tgt_mask, tgt_key_padding_mask = self.make_tgt_mask(tgt)

        # Encode
        memory = self.encode(src, src_key_padding_mask)
        
        # Decode
        decoder_output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src,
        max_len=3000,
        sos_id=1,
        eos_id=2,
        temperature=1.0,
        top_k=None,
        top_p=None,
        force_rest_every=None,  # New parameter
        time_shift_48_id=None   # New parameter - pass vocab['TIME_SHIFT_48']
    ):
        """
        Autoregressive generation with KV caching.
        """
        self.eval()
        device = src.device
        
        # Encode source (only once)
        src_key_padding_mask = self.make_src_mask(src)
        memory = self.encode(src, src_key_padding_mask)
        
        # Start with SOS
        generated = [sos_id]
        
        # Initialize KV cache: list of (k, v) tuples for each decoder layer
        # Each k, v has shape (batch, num_heads, seq_len, head_dim)
        kv_cache = [None] * self.transformer.decoder.num_layers
        
        # Process first token to initialize cache
        tgt = torch.tensor([[sos_id]], device=device)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pos(tgt_emb)
        
        # Run through decoder layers manually to build cache
        x = tgt_emb
        for i, layer in enumerate(self.transformer.decoder.layers):
            x, kv_cache[i] = self._decoder_layer_with_cache(
                layer, x, memory, 
                memory_key_padding_mask=src_key_padding_mask,
                kv_cache=None,  # No cache for first token
                pos_idx=0
            )
        
        if self.transformer.decoder.norm is not None:
            x = self.transformer.decoder.norm(x)
        
        note_count = 0  # Track number of notes generated
        
        for step in range(max_len - 1):
            # Get logits for last position
            logits = self.output_proj(x[:, -1:, :]).squeeze(1)  # (1, vocab_size)
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Debug
            print(f"Step {step}: token={next_token}")
            
            # Force TIME_SHIFT_48 every N notes
            if (force_rest_every is not None and 
                time_shift_48_id is not None and 
                note_count > 0 and 
                note_count % force_rest_every == 0):
                next_token = time_shift_48_id
                print(f"Forced TIME_SHIFT_48 at note {note_count}")
            
            generated.append(next_token)
            
            # Count notes (tokens starting with NOTE_)
            # We need to check what token this ID corresponds to
            # For simplicity, just increment if it's likely a note based on generation pattern
            # Better: pass id_to_token mapping
            if next_token == eos_id:
                break
            
            # Increment note count after TIME_SHIFT (since pattern is TIME_SHIFT -> NOTE -> DURATION)
            # This is a rough heuristic - you may want to track actual NOTE tokens
            note_count += 1
                
            # Process only the new token (this is where KV cache saves computation)
            new_tgt = torch.tensor([[next_token]], device=device)
            new_tgt_emb = self.decoder_embedding(new_tgt) * math.sqrt(self.d_model)
            
            # Add positional encoding for the current position
            pos_idx = len(generated) - 1
            new_tgt_emb = new_tgt_emb + self.decoder_pos.pe[:, pos_idx:pos_idx+1, :]
            
            # Run through decoder layers with cache
            x = new_tgt_emb
            for i, layer in enumerate(self.transformer.decoder.layers):
                x, kv_cache[i] = self._decoder_layer_with_cache(
                    layer, x, memory,
                    memory_key_padding_mask=src_key_padding_mask,
                    kv_cache=kv_cache[i],
                    pos_idx=pos_idx
                )
            
            if self.transformer.decoder.norm is not None:
                x = self.transformer.decoder.norm(x)
        
        return generated

    def _decoder_layer_with_cache(self, layer, x, memory, memory_key_padding_mask=None, kv_cache=None, pos_idx=0):
        """
        Run a single decoder layer with KV caching.
        
        Args:
            layer: TransformerDecoderLayer
            x: input tensor (batch, 1, d_model) for cached, (batch, seq, d_model) for first
            memory: encoder output
            kv_cache: tuple of (k, v) from previous steps, or None
            pos_idx: current position index
        
        Returns:
            output: (batch, 1, d_model)
            new_cache: updated (k, v) tuple
        """
        # Self-attention with cache
        # norm_first=True means we normalize before attention
        residual = x
        x = layer.norm1(x)
        
        # Compute Q, K, V for current token(s)
        # We need to access the self_attn's in_proj to get Q, K, V
        bsz, tgt_len, embed_dim = x.shape
        
        # Get Q, K, V projections
        # in_proj_weight is (3*embed_dim, embed_dim)
        # in_proj_bias is (3*embed_dim,)
        qkv = F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        head_dim = embed_dim // layer.self_attn.num_heads
        num_heads = layer.self_attn.num_heads
        
        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        
        # Concatenate with cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        # Store updated cache
        new_cache = (k, v)
        
        # Compute attention (causal - only attend to past)
        # No explicit mask needed since we only have past keys
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Causal is implicit - we only have past K,V
            dropout_p=0.0,
            is_causal=False  # We handle causality via the cache structure
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        
        # Output projection
        attn_output = layer.self_attn.out_proj(attn_output)
        x = residual + attn_output
        
        # Cross-attention
        residual = x
        x = layer.norm2(x)
        x = residual + layer.multihead_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )[0]
        
        # FFN
        residual = x
        x = layer.norm3(x)
        x = residual + layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        
        return x, new_cache