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
    int spaceCount = 0;
    
    cerr<<"Enter a string: ";   // cerr is the standard error stream, cerr writes output immediately without buffering.
    getline(cin, input);
    
    for(char c : input){
        if(isspace(c)){
            spaceCount++;
        }
    }
    
    cout<<"Number of white spaces: "<<spaceCount<<nl;
}

signed main(){
    FastIO();
    solution();
    return 0;
}