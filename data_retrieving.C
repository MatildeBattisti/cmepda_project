#include <iostream>
#include <TFile.h>
#include <TTree.h>

void data_retrieving() {
    /**
     * @brief Open file from local path.
     * Returns an error if file is not opened.
     */
    //TFile *file = TFile::Open("/home/matilde/Downloads/1C037D8D-2092-2448-81A4-BE32B05BFB45.root");
    TFile *file = TFile::Open("/home/matilde/Documenti/cmepda_project/skimmed.root");

    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file." << std::endl;
        exit(-1);
     }
    file->ls();
    
    /**
     * @brief Get 'Events' tree entries.
     * Default -> EVENT:-1. We only use it to see the entries name.
     * Returns error if the tree isn't loaded.
     */
    TTree *event = (TTree*)file->Get("Events");

    if (!event) {
        std::cerr << "Error loading TTree 'Events'" << std::endl;
        file->Close();
        exit(-1);
    }

    event->Show();

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