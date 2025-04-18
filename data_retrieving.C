#include <iostream>
#include <TFile.h>
#include <TTree.h>

void data_retrieving() {
    /**
     * @brief Open file from local path.
     * Returns an error if file is not opened.
     */
    //TFile *file = TFile::Open("datasets/2E96A5E9-C938-A149-BBBF-8FD81A9E5AD6.root");
    TFile *file = TFile::Open("datasets/skimmed.root");

    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file." << std::endl;
        exit(-1);
     }
    file->ls();
    
    /**
     * @brief Get 'Events' tree entries.
     * If Show() is left empty, shows EVENT:-1. We only use it to see the entries name.
     * Returns error if the tree isn't loaded.
     */
    TTree *event = (TTree*)file->Get("Events");

    if (!event) {
        std::cerr << "Error loading TTree 'Events'" << std::endl;
        file->Close();
        exit(-1);
    }

    event->Show(1);

    /**
     * @brief Gets number of branches inside the 'Events' TTree.
     */
    int nBranches = event->GetListOfBranches()->GetEntries();
    std::cout << "Number of branches in Events: " << nBranches << std::endl;

    /**
     * @brief Closes file.
     */
    file->Close();
}