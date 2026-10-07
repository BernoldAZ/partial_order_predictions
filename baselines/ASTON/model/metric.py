#Damerau-Levenshtein measure to determine similarity between traces 
def levenshtein_similarity(pred_trace, ground_truth_trace, code_end):
    try:
        l1 = pred_trace.index(code_end)
    except ValueError:
        l1 = len(pred_trace)
    try:
        l2 = ground_truth_trace.index(code_end)
    except ValueError:
        l2 = len(pred_trace)   
    if max(l1, l2) == 0: return 0, 0, 1.0
    matrix = [list(range(l1 + 1))] * (l2 + 1)
    for zz in list(range(l2 + 1)):
      matrix[zz] = list(range(zz,zz + l1 + 1))
    for zz in list(range(0,l2)):
      for sz in list(range(0,l1)):
        if pred_trace[sz] == ground_truth_trace[zz]:
          matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
        else:
          matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)
    distance = float(matrix[l2][l1])
    result = 1.0-distance/max(l1,l2)
    return l1, l2, result
