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

// Function to remove whitespace from a string (returns a new string)
string removeWhitespace(const string &str){
    string result;
    for(char ch : str){
        if(!isspace(ch)){
            result += ch;
        }
    }
    return result;
}

void solution(){
    string input;
    cerr<<"Enter a string: ";
    getline(cin, input);

    input = removeWhitespace(input);

    cout<<"Output: "<<input<<endl;
}

signed main() {
    FastIO();
    solution();
    return 0;
}
