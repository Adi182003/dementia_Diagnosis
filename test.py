def solve(threshold, points):
    def recursion(points, low, high, key, ans):
        mid=(low+high)//2
        if points[mid]==key:
            ans=mid
            return
        elif points[mid]>=key:
            ans=min(ans, mid)
            recursion(points, low, mid-1, key, ans)
        else:
            recursion(points, mid+1, high, key, ans)
            
            
    
    if (threshold+points[0]) > (points[-1]):
        return len(points)
    key=threshold + points[0]
    ans=len(points)-1
    recursion(points, 0, len(points)-1, key, ans)
    ans+=1
    res=(ans+2)//2
    return res
    
threshold=2
points=[1,2,3]
print(solve(threshold, points))