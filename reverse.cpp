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

void solution() {
    string str;
    cerr<<"Enter a string: ";
    getline(cin, str);

    cout<<"Characters in reverse order:"<<endl;
    for(int i = str.length() - 1; i >= 0; i--) {
        cout << str[i];
    }
    cout<<nl;
}

signed main() {
    FastIO();
    solution();
    return 0;
}
