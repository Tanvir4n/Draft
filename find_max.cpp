#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>

using namespace std;

string findMostFrequentWord(const string& sentence) {
    unordered_map<string, int> wordCount;
    istringstream iss(sentence);
    string word;
    
    // Count word occurrences
    while (iss >> word) {
        // Remove punctuation and make lowercase
        word.erase(remove_if(word.begin(), word.end(), ::ispunct), word.end());
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        wordCount[word]++;
    }
    
    // Find most frequent word
    string mostFrequent;
    int maxCount = 0;
    for (const auto& pair : wordCount) {
        if (pair.second > maxCount) {
            mostFrequent = pair.first;
            maxCount = pair.second;
        }
    }
    
    return mostFrequent;
}

int main() {  // Make sure this is 'main' not 'WinMain'
    string input;
    cout << "Enter a sentence: ";
    getline(cin, input);
    
    string result = findMostFrequentWord(input);
    cout << "Most frequent word: " << result << endl;
    
    return 0;
}