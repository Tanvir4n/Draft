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

void FastIO() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
}

void solution(){
    string str;
    int alphabets = 0, digits = 0, special = 0;

    cout << "Enter a string: ";
    getline(cin, str);

    for (char ch : str){
        if (isalpha(ch)){
            alphabets++;
        } else if (isdigit(ch)){
            digits++;
        } else if (!isspace(ch)){
            special++;
        }
    }

    cout << "Alphabets: " << alphabets << endl;
    cout << "Digits: " << digits << endl;
    cout << "Special characters: " << special << endl;
}

signed main() {
    FastIO();
    solution();
    return 0;
}


/*
#include <iostream>
#include <string>
#include <cctype>

using namespace std;

int main() {
    string str;
    int alphabets = 0, digits = 0, special = 0;

    cout << "Enter a string: ";
    getline(cin, str);  // Reads the full line including spaces

    for (char ch : str) {
        if (isalpha(ch)) {
            alphabets++;
        } else if (isdigit(ch)) {
            digits++;
        } else if (!isspace(ch)) {
            special++;
        }
    }

    cout << "Alphabets: " << alphabets << endl;
    cout << "Digits: " << digits << endl;
    cout << "Special characters: " << special << endl;

    return 0;
}

*/