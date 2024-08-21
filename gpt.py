import jax
import jax.numpy as jnp
from jax import lax

def pad_to_fixed_size(arr, max_size, pad_value=0):
    """Pad an array to the max_size along the first dimension."""
    pad_length = max_size - arr.shape[0]
    return jnp.pad(arr, ((0, pad_length), (0, 0)), constant_values=pad_value)

def process_block(ratings, matches):
    def update_fn(rating, match):
        # Apply the update only if match is valid (i.e., not a padding entry)
        return lax.cond(
            match[0] == -1,  # Condition for padding
            lambda x: rating,  # If padding, return unchanged rating
            lambda match: rating.at[match[1:]].add(1),  # Otherwise, perform update
            match
        )
    
    # Apply the update function over all matches in the block
    updated_ratings = jax.lax.fori_loop(0, matches.shape[0], lambda i, r: update_fn(r, matches[i]), ratings)
    return updated_ratings

def scan_fn(carry, block):
    ratings, _ = carry
    # Get the matches within this block
    matches = block[1:]
    updated_ratings = process_block(ratings, matches)
    return (updated_ratings, None), None

# Parameters
block_size = 5  # Number of time periods in a block

# Example schedule array (N, 3)
schedule = jnp.array([
    [0, 1, 2],
    [0, 3, 4],
    [1, 1, 3],
    [2, 2, 4],
    [2, 3, 5],
])

# Ratings array
ratings = jnp.zeros(jnp.max(schedule[:, 1:]) + 1)

# Split into blocks and pad each block
blocks = []
unique_times = jnp.unique(schedule[:, 0])
num_blocks = (len(unique_times) + block_size - 1) // block_size

for i in range(num_blocks):
    block_start = i * block_size
    block_end = min((i + 1) * block_size, len(unique_times))
    block_times = unique_times[block_start:block_end]

    # Gather matches for the current block
    matches_in_block = []
    for time in block_times:
        matches = schedule[schedule[:, 0] == time]
        matches_in_block.append(matches)

    # Pad to make a consistent block size
    max_matches_in_block = max([len(m) for m in matches_in_block])
    padded_matches = [pad_to_fixed_size(m, max_matches_in_block, pad_value=-1) for m in matches_in_block]
    
    block = jnp.concatenate(padded_matches)
    blocks.append(block)

blocks = jnp.array(blocks)

# Initial carry
initial_carry = (ratings, None)

# Use scan to process each block
final_carry, _ = lax.scan(scan_fn, initial_carry, blocks)
final_ratings = final_carry[0]

print(final_ratings)
