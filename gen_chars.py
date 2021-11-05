def gen_chars(eye, slic):
    """
    Recibe la imagen y el SLIC y retorna las entradas a la red
    [N, 5] = [[x, y, r, g, b],
                ...]
    
    """
    out = []
    h, w = slic.shape
    upper_bound = 0
    lower_bound = h-1
    right_bound = w-1
    left_bound = 0
    slic = np.array(slic)
    bo = slic>0
    while 1:
        if np.any(bo[:, left_bound]):
            break
        left_bound += 1
    while 1:
        if np.any(bo[:, right_bound]):
            break
        right_bound -= 1
    while 1:
        if np.any(bo[upper_bound, :]):
            break
        upper_bound += 1
    while 1:
        if np.any(bo[lower_bound, :]):
            break
        lower_bound -= 1
    print(upper_bound, lower_bound, left_bound, right_bound) 
    for i in range(1, len(np.unique(slic))):
        ix, jx = np.where(slic==i)
        px = np.array([ix, jx]).T
        mpx = np.mean(px, 0)
        
        rgbx = np.mean(eye[slic==i], 0)/255
        out.append([(mpx[0]-upper_bound)/(lower_bound-upper_bound),
                   (mpx[1]-left_bound)/(right_bound-left_bound),
                   rgbx[0], rgbx[1], rgbx[2]])
    return np.array(out)
        
    