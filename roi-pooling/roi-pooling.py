import numpy as np

def roi_pool(feature_map, rois, output_size):
    fm = np.array(feature_map)
    results = []
    
    for (x1, y1, x2, y2) in rois:
        roi_h = y2 - y1
        roi_w = x2 - x1
        out = np.zeros((output_size, output_size), dtype=fm.dtype)
        
        for i in range(output_size):
            for j in range(output_size):
                hstart = y1 + int(i * roi_h / output_size)
                hend   = y1 + int((i + 1) * roi_h / output_size)
                wstart = x1 + int(j * roi_w / output_size)
                wend   = x1 + int((j + 1) * roi_w / output_size)
                
                if hend == hstart: hend = hstart + 1
                if wend == wstart: wend = wstart + 1
                
                out[i, j] = fm[hstart:hend, wstart:wend].max()
        
        results.append(out.tolist())
    
    return results