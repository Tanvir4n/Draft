#include <bits/stdc++.h>

using namespace std;

#define int long long
#define ll long long
#define ull unsigned long long
#define ui unsigned int
#define pi acos(-1)
#define pb push_back
#define endl "\n"
#define nl "\n"

void FastIO(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
}

void solution(){
    char input[1000], copied[1000];
    cout << "Enter a line: ";
    cin.getline(input, 1000);

    // Manually copy input to copied
    int i = 0;
    while (input[i] != '\0') {
        copied[i] = input[i];
        i++;
    }
    copied[i] = '\0';

    // Tokenize using stringstream
    map<string, int> freq;
    string word;
    stringstream ss(copied);

    while (ss >> word) {
        // Optional: remove punctuation
        string clean = "";
        for (char ch : word) {
            if (isalpha(ch))
                clean += tolower(ch); // Normalize to lowercase
        }
        if (!clean.empty())
            freq[clean]++;
    }

    // Find max frequency word
    string maxWord;
    int maxFreq = 0;

    for (auto it : freq) {
        if (it.second > maxFreq) {
            maxFreq = it.second;
            maxWord = it.first;
        }
    }

    cout << "Most frequent word: '" << maxWord << "' appears " << maxFreq << " times." << endl;
}

signed main(){
    FastIO();
    solution();
    return 0;
}
