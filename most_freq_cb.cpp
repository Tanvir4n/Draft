#include <iostream>
#include <string>
#include <map>
#include <sstream>

using namespace std;

string findMostFrequentWord(const string& sentence) {
    map<string, int> wordCount;
    istringstream iss(sentence);
    string word;

    while (iss >> word) {
        wordCount[word]++;
    }

    string result;
    int maxCount = 0;
    for (const auto& pair : wordCount) {
        if (pair.second > maxCount) {
            result = pair.first;
            maxCount = pair.second;
        }
    }

    return result;
}
