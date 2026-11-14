import java.util.*;

public class VowelSubstrings {
    private static final Set<Character> VOWELS = Set.of('a', 'e', 'i', 'o', 'u');

    public static int countVowelSubstrings(String s) {
        int n = s.length();
        int total = 0;
        int i = 0;

        while (i < n) {
            // skip consonants
            if (!VOWELS.contains(s.charAt(i))) {
                i++;
                continue;
            }

            // start of vowel segment
            int j = i;
            while (j < n && VOWELS.contains(s.charAt(j))) {
                j++;
            }

            // process segment [i, j-1]
            total += countInSegment(s.substring(i, j));
            i = j;
        }

        return total;
    }

    private static int countInSegment(String segment) {
        int n = segment.length();
        int[] freq = new int[26];
        int unique = 0;
        int left = 0;
        int total = 0;

        for (int right = 0; right < n; right++) {
            char c = segment.charAt(right);
            if (freq[c - 'a']++ == 0)
                unique++;

            while (unique == 5) {
                // all vowels are present
                total += (n - right); // all substrings [left..end] valid
                char leftChar = segment.charAt(left++);
                if (--freq[leftChar - 'a'] == 0)
                    unique--;
            }
        }
        return total;
    }

    // --- Testing ---
    public static void main(String[] args) {
        System.out.println(countVowelSubstrings("aeioaexaaeuiou")); // 4
        System.out.println(countVowelSubstrings("aaeiouxa")); // 2
        System.out.println(countVowelSubstrings("aeiou")); // 1
        System.out.println(countVowelSubstrings("abc")); // 0
        System.out.println(countVowelSubstrings("aaaeeeiioouuu")); // many
    }
}
