#include <bits/stdc++.h>
using namespace std;

#define int long long
#define endl "\n"

void FastIO() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
}

void solution() {
    char original[100], copied[100];
    string line;

    cerr << "Enter a line: ";
    cin.getline(original, 100);

    // 1. Manual copy
    int i = 0;
    while (original[i] != '\0') {
        copied[i] = original[i];
        i++;
    }
    copied[i] = '\0';

    cout << "\nCopied string: " << copied << endl;

    // Convert char array to string
    line = string(copied);

    // 2. Tokenize and process
    string word;
    stringstream ss(line);
    unordered_map<string, int> freq;
    vector<string> tokens;
    //vector<pair<string, int>>
    int articleCount = 0;

    while (ss >> word) {
        // Remove punctuation
        word.erase(remove_if(word.begin(), word.end(), ::ispunct), word.end());
        // Convert to lowercase
        transform(word.begin(), word.end(), word.begin(), ::tolower);

        if (word == "a" || word == "an" || word == "the") {
            articleCount++;
        }

        if (!word.empty()) {
            freq[word]++;
            tokens.push_back(word);
        }
    }

    // 3. Find most frequent word
    string mostFrequent;
    int maxCount = 0;
    for (auto& p : freq) {
        if (p.second > maxCount) {
            maxCount = p.second;
            mostFrequent = p.first;
        }
    }

    // Output
    cout << "Total articles found: " << articleCount << endl;
    cout << "Most frequent word: " << mostFrequent << " (appeared " << maxCount << " times)" << endl;

    cout << "All tokenized words:" << endl;
    for (const string& w : tokens) {
        cout << w << endl;
    }
}

signed main() {
    FastIO();
    solution();
    return 0;
}
