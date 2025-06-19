#include <iostream>
#include <string>
using namespace std;

int main() {
    string text;
    int count = 0;
    
    cout << "Enter your text: ";
    getline(cin, text); // Take input from user
    
    // Check each word in the text
    string word = "";
    for (char c : text) {
        if (isalpha(c)) {
            word += tolower(c); // Convert to lowercase and build word
        } else {
            // Check if the word is an article
            if (word == "a" || word == "an" || word == "the") {
                count++;
            }
            word = ""; // Reset for next word
        }
    }
    
    // Check the last word in case text ends without punctuation
    if (word == "a" || word == "an" || word == "the") {
        count++;
    }
    
    cout << "Total articles found: " << count << endl;
    
    return 0;
}