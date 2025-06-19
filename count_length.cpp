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
    string input;
    
    cout<<"Enter a string: ";
    cout.flush();  // Force the output to be displayed
    
    getline(cin, input);
    
    int length = input.length();
    cout<<"Length of the string: "<<length<<nl;
}

signed main(){
    FastIO();
    solution();
    return 0;
}