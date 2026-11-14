import java.util.*;

public class PhoneCallsMaxVolume {

    static class Call {
        long start;
        long end; // treat as CLOSED: [start, end] w.r.t. overlap rule (i.e., end == next start =>
                  // overlap)
        int vol;

        Call(long s, long e, int v) {
            this.start = s;
            this.end = e;
            this.vol = v;
        }
    }

    /**
     * Returns the maximum total order volume from non-overlapping calls.
     * Overlap rule: a call that starts at time T overlaps any call that ends at
     * time T.
     * i.e., to chain calls i -> j, we require end[i] < start[j] (STRICT).
     */
    public static int phoneCalls(List<Integer> start,
            List<Integer> duration,
            List<Integer> volume) {
        if (start == null || duration == null || volume == null)
            throw new IllegalArgumentException("Inputs must not be null");
        int n = start.size();
        if (duration.size() != n || volume.size() != n)
            throw new IllegalArgumentException("All lists must have the same size");
        if (n == 0)
            return 0;

        // Build and sort by end time
        Call[] calls = new Call[n];
        for (int i = 0; i < n; i++) {
            long s = start.get(i);
            long e = s + (long) duration.get(i); // use long to avoid overflow
            calls[i] = new Call(s, e, volume.get(i));
        }
        Arrays.sort(calls, Comparator.comparingLong(c -> c.end));

        long[] ends = new long[n];
        for (int i = 0; i < n; i++)
            ends[i] = calls[i].end;

        // DP: dp[i] = best total volume using calls up to index i (inclusive)
        long[] dp = new long[n];
        dp[0] = calls[0].vol;

        for (int i = 1; i < n; i++) {
            // STRICT compatibility: find last j < i with ends[j] < calls[i].start
            int p = lowerBound(ends, calls[i].start) - 1; // first >= start, then step back

            long take = calls[i].vol + (p >= 0 ? dp[p] : 0);
            long skip = dp[i - 1];
            dp[i] = Math.max(take, skip);
        }

        return (int) dp[n - 1]; // safe per constraints; dp uses long for headroom
    }

    // lower_bound: first index with arr[idx] >= key
    private static int lowerBound(long[] arr, long key) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (arr[mid] >= key)
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }

    // Quick sanity checks
    public static void main(String[] args) {
        // Your new example: answer should be 4
        List<Integer> start1 = Arrays.asList(1, 2, 4);
        List<Integer> dura1 = Arrays.asList(2, 2, 1);
        List<Integer> vol1 = Arrays.asList(1, 2, 3);
        System.out.println(phoneCalls(start1, dura1, vol1)); // 4

        // Previous example: answer should be 76 (calls 2 and 4)
        List<Integer> start2 = Arrays.asList(10, 5, 15, 18, 30);
        List<Integer> dura2 = Arrays.asList(30, 12, 20, 35, 35);
        List<Integer> vol2 = Arrays.asList(50, 51, 20, 25, 10);
        System.out.println(phoneCalls(start2, dura2, vol2)); // 76
    }
}
