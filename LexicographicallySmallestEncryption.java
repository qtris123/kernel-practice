import java.util.*;

public class LexicographicallySmallestEncryption {
    public static String encrypt(String original) {
        int n = original.length();
        if (n == 0)
            return "";

        char[] s = original.toCharArray();

        // 1) Precompute minSuffix[i] = smallest char in s[i..n-1]
        char[] minSuffix = new char[n];
        char curMin = s[n - 1];
        minSuffix[n - 1] = curMin;
        for (int i = n - 2; i >= 0; --i) {
            curMin = (char) Math.min(curMin, s[i]);
            minSuffix[i] = curMin;
        }

        // 2) Simulate with a stack for temporaryString
        Deque<Character> stack = new ArrayDeque<>();
        StringBuilder result = new StringBuilder(n);

        int i = 0; // pointer in original
        while (i < n || !stack.isEmpty()) {
            if (stack.isEmpty()) {
                // Must push if nothing to pop
                stack.addLast(s[i++]);
            } else {
                // If input remains, compare stack top vs min remaining
                char top = stack.peekLast();
                char minRem = (i < n) ? minSuffix[i] : Character.MAX_VALUE;

                if (top <= minRem) {
                    // Popping now yields the smallest lexicographic result
                    result.append(stack.removeLast());
                } else {
                    // There exists a smaller char ahead; push more
                    stack.addLast(s[i++]);
                }
            }
        }

        return result.toString();
    }

    // Quick demo
    public static void main(String[] args) {
        System.out.println(encrypt("abc")); // bdy (given example)
        System.out.println(encrypt("vgxgpu")); // ""
        // System.out.println(encrypt("a")); // "abcd"
        // System.out.println(encrypt("zyx")); // "xyz"
        // System.out.println(encrypt("bbab")); // "abbb"
        // System.out.println(encrypt("cabbacd")); // "aabbccd"
    }
}
