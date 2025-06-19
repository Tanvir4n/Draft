#include <bits/stdc++.h>
using namespace std;

void FastIO() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
}

void solution() {
    string line;
    getline(cin, line);

    // Convert whole line to lowercase for case-insensitive matching
    transform(line.begin(), line.end(), line.begin(), ::tolower);

    // Use stringstream to split the line into words
    stringstream ss(line);
    string word;

    int countA = 0, countAn = 0, countThe = 0;

    while (ss >> word) {
        if (word == "a") countA++;
        else if (word == "an") countAn++;
        else if (word == "the") countThe++;
    }

    int total = countA + countAn + countThe;

    cout << "Total articles count: " << total << endl;
    cout << "\"a\" appears " << countA << " times" << endl;
    cout << "\"an\" appears " << countAn << " times" << endl;
    cout << "\"the\" appears " << countThe << " times" << endl;
}

signed main() {
    FastIO();
    solution();
    return 0;
}
