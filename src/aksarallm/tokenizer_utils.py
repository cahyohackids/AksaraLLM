"""
AksaraLLM Tokenizer Utilities.

Wrapper around the BPE tokenizer with proper special token handling.
"""

import torch
from typing import List, Optional, Union

__all__ = ["AksaraTokenizer"]


class AksaraTokenizer:
    """Wrapper around the AksaraLLM BPE tokenizer.
    
    Handles special token edge cases (auto-BOS/EOS stripping)
    and provides a clean encode/decode interface.
    
    Example:
        >>> tok = AksaraTokenizer()
        >>> ids = tok.encode("Halo dunia!")
        >>> text = tok.decode(ids)
    """
    
    def __init__(self, repo_id: str = "AksaraLLM/aksara-tokenizer-v3"):
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        
        tok_path = hf_hub_download(repo_id, "tokenizer.json")
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.no_padding()
        self._tok.no_truncation()
        
        self.eos_id = self._tok.token_to_id("[EOS]") or 1
        self.bos_id = self._tok.token_to_id("[BOS]") or 2
        self.pad_id = self._tok.token_to_id("[PAD]") or 0
        self.unk_id = self._tok.token_to_id("[UNK]") or 3
        self.vocab_size = self._tok.get_vocab_size()
    
    def encode(
        self, 
        text: str, 
        add_bos: bool = False, 
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Prepend BOS token
            add_eos: Append EOS token
            max_length: Optional maximum length (truncate if exceeded)
        """
        ids = self._tok.encode(text).ids
        
        # Strip auto-added special tokens (tokenizer sometimes adds them)
        if ids and ids[0] == self.bos_id:
            ids = ids[1:]
        if ids and ids[-1] == self.eos_id:
            ids = ids[:-1]
        
        # Truncate if needed (before adding specials)
        if max_length is not None:
            reserve = int(add_bos) + int(add_eos)
            ids = ids[:max_length - reserve]
        
        # Add special tokens
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(
        self, 
        ids: Union[List[int], torch.Tensor], 
        skip_special: bool = True,
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            ids: Token IDs (list or tensor)
            skip_special: Skip special tokens in output
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        
        # Filter out padding
        ids = [i for i in ids if i != self.pad_id]
        
        if not ids:
            return ""
        
        return self._tok.decode(ids, skip_special_tokens=skip_special)
    
    def batch_encode(
        self, 
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> dict:
        """Encode a batch of texts.
        
        Args:
            texts: List of text strings
            add_bos: Prepend BOS token  
            add_eos: Append EOS token
            max_length: Maximum length per sequence
            padding: Pad all sequences to same length
            
        Returns:
            dict with 'input_ids' and optionally 'attention_mask'
        """
        all_ids = [self.encode(t, add_bos, add_eos, max_length) for t in texts]
        
        if not padding:
            return {"input_ids": all_ids}
        
        # Pad to longest sequence
        max_len = max(len(ids) for ids in all_ids)
        padded = []
        masks = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [self.pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        }
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID."""
        return self._tok.token_to_id(token)
    
    def id_to_token(self, id: int) -> Optional[str]:
        """Convert a token ID to its string."""
        return self._tok.id_to_token(id)
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return (
            f"AksaraTokenizer(vocab_size={self.vocab_size}, "
            f"bos={self.bos_id}, eos={self.eos_id}, pad={self.pad_id})"
        )
