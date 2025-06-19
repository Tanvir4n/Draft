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

void compareStrings(const string &str1, const string &str2){
    if(str1 == str2){
        cout << "Strings are equal" << endl;
    } else if(str1 < str2){
        cout<< "First string is less than second string" << endl;
    } else {
        cout << "First string is greater than second string" << endl;
    }
}

void solution(){
    string str1, str2;

    cerr<<"First string: ";
    getline(cin, str1);

    cerr<< "Second string: ";
    getline(cin, str2);

    compareStrings(str1, str2);
}

signed main(){
    FastIO();
    solution();
    return 0;
}
