def getMedianUnmanaged(a):
    low = 0
    high = len(a) - 1
    median = (low + high) / 2
    while True:
        if high < 0:
            return 0
        if high <= low:
            return a[median]
        if high == low + 1:
            if a[low] > a[high]:
                a[low],a[high] = a[high],a[low]
            return a[median]
        middle = (low+high)/2
        if a[middle] > a[high]:
            a[middle],a[high] = a[high],a[middle]
        if a[low] > a[high]:
            a[low],a[high] = a[high],a[low]
        if a[middle] > a[low]:
            a[middle],a[low] = a[low],a[middle]
        a[middle],a[low+1] = a[low+1],a[middle]
        
        ll = low + 1
        hh = high
        
        while True:
            while True:
                ll +=1
                try:
                    if a[low] > a[ll]:
                        break                    
                except IndexError:
                    break
            while True:
                hh -= 1
                try:
                    if a[hh] > a[low]:
                        break
                except IndexError:
                    break
            if hh < ll:
                break
            try:
                a[ll],a[hh] = a[hh],a[ll]
            except IndexError:
                break
        try:
            a[low],a[hh] = a[hh],a[low]
        except IndexError:
            break
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1

def getMedian(a):
    median = getMedianUnmanaged(a)
    return median

