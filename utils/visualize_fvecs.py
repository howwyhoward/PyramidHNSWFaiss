#!/usr/bin/env python3
# To run this script:
# python utils/visualize_fvecs.py data/siftsmall/siftsmall_base.fvecs
# python utils/visualize_fvecs.py data/siftsmall/siftsmall_groundtruth.ivecs 
# 
# Arguments:
#   1. Path to the .fvecs or .ivecs file
#   2. (Optional) Maximum bytes to read for visualization (default: 2048)

import os
import struct
import sys
import binascii

def visualize_binary_file(filename, max_bytes=2048):
    """
    Create a visual representation of a binary file structure,
    showing how data is organized in bytes.
    """
    file_size = os.path.getsize(filename)
    is_fvecs = filename.endswith('.fvecs')
    is_ivecs = filename.endswith('.ivecs')
    
    print(f"File: {filename}")
    print(f"Size: {file_size} bytes")
    
    if is_fvecs:
        print("Type: FVECS (Floating-point vectors)")
    elif is_ivecs:
        print("Type: IVECS (Integer vectors)")
    else:
        print("Type: Unknown binary format")
    
    print("\nBinary Structure Visualization:")
    print("================================")
    
    with open(filename, 'rb') as f:
        # Read first few bytes
        data = f.read(min(max_bytes, file_size))
        
        # Determine how many vectors we can visualize
        vectors = []
        offset = 0
        
        while offset < len(data):
            if offset + 4 > len(data):
                break
                
            # Read dimension/count (first 4 bytes)
            dim = struct.unpack('i', data[offset:offset+4])[0]
            
            # Quick validation - SIFT is 128 dimensions
            if is_fvecs and dim != 128 and not (80 <= dim <= 200):
                # This might be endian issues, try with big endian
                dim = struct.unpack('>i', data[offset:offset+4])[0]
                if dim != 128 and not (80 <= dim <= 200):
                    print(f"Warning: Expected dimension around 128 but found {dim} at offset {offset}")
                    # Skip this byte and try to resync
                    offset += 1
                    continue
            
            offset += 4
            
            # Check if we have complete vector data
            if offset + dim*4 > len(data):
                # Incomplete vector
                break
                
            # Extract vector data
            vector_data = data[offset:offset+dim*4]
            offset += dim*4
            
            if is_fvecs:
                # Parse floats
                values = struct.unpack(f'{dim}f', vector_data)
                vectors.append((dim, values))
            elif is_ivecs:
                # Parse integers
                values = struct.unpack(f'{dim}i', vector_data)
                vectors.append((dim, values))
            else:
                # Just store raw bytes
                vectors.append((dim, vector_data))
        
        # Visualize the structure
        print(f"Found {len(vectors)} complete vectors in the preview")
        
        offset = 0
        for i, (dim, values) in enumerate(vectors):
            print(f"\nVector {i}:")
            print(f"  Offset: {offset} bytes")
            print(f"  Dimension: {dim}")
            
            # Dimension/count bytes
            dim_bytes = struct.pack('i', dim)
            dim_hex = binascii.hexlify(dim_bytes).decode()
            print(f"  Dimension Header: {dim_hex} (hex) = {dim} (int)")
            
            # Show byte layout
            print("  Layout:")
            print("  ┌───────────────────────────────┐")
            
            # Dimension header row
            print(f"  │ {' '.join([dim_hex[j:j+2] for j in range(0, len(dim_hex), 2)])} │ Dimension/Count")
            print("  ├───────────────────────────────┤")
            
            # Show first few values
            max_show = min(dim, 5)
            for j in range(max_show):
                if is_fvecs:
                    val = values[j]
                    val_bytes = struct.pack('f', val)
                    val_hex = binascii.hexlify(val_bytes).decode()
                    print(f"  │ {' '.join([val_hex[k:k+2] for k in range(0, len(val_hex), 2)])} │ Value[{j}]: {val}")
                elif is_ivecs:
                    val = values[j]
                    val_bytes = struct.pack('i', val)
                    val_hex = binascii.hexlify(val_bytes).decode()
                    print(f"  │ {' '.join([val_hex[k:k+2] for k in range(0, len(val_hex), 2)])} │ Value[{j}]: {val}")
            
            if dim > max_show:
                print("  │           ...               │")
                # Show last value
                if is_fvecs or is_ivecs:
                    val = values[dim-1]
                    if is_fvecs:
                        val_bytes = struct.pack('f', val)
                    else:
                        val_bytes = struct.pack('i', val)
                    val_hex = binascii.hexlify(val_bytes).decode()
                    print(f"  │ {' '.join([val_hex[k:k+2] for k in range(0, len(val_hex), 2)])} │ Value[{dim-1}]: {val}")
            
            print("  └───────────────────────────────┘")
            
            if is_fvecs:
                print(f"  Summary: Vector with {dim} float values")
                print(f"  Range: Min={min(values):.2f}, Max={max(values):.2f}, Mean={sum(values)/len(values):.2f}")
            elif is_ivecs:
                print(f"  Summary: Vector with {dim} integer values")
                print(f"  Range: Min={min(values)}, Max={max(values)}, Mean={sum(values)/len(values):.2f}")
            
            # Update offset for next vector
            offset += 4 + dim*4
            
            # Limit number of vectors displayed
            if i >= 2:
                print("\n... (more vectors available) ...")
                break

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename.fvecs|filename.ivecs> [max_bytes]")
        return
    
    filename = sys.argv[1]
    max_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    
    visualize_binary_file(filename, max_bytes)

if __name__ == "__main__":
    main() 