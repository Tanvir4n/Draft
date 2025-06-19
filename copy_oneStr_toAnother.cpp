#include <iostream>
#include <string>
using namespace std;

void solution() {
    string source, destination;
    cout << "Enter the source string: ";
    getline(cin, source);

    // Copy the source string to destination
    destination = source;

    cout << "Copied string: " << destination << endl;
}

int main() {
    solution();
    return 0;
}


/*
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
    char str1[100], str2[100];

    cout << "Enter a string: ";
    cin.getline(str1, 100);

    // Copy manually without any function
    int i = 0;
    while(str1[i] != '\0'){
        str2[i] = str1[i];
        i++;
    }
    str2[i] = '\0'; // Add null terminator

    cout << "Copied string: " << str2 << endl;
}
 
signed main(){
    FastIO();
    solution();
    return 0;
}

*/