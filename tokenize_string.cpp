#include <bits/stdc++.h>
using namespace std;

void solution(){
    string line;
    getline(cin, line);

    stringstream ss(line);
    string word;
    while (ss >> word){
        cout << word << "\n";
    }
}

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    solution();

    return 0;
}
