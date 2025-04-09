#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TObjArray.h>
#include <TString.h>
#include <iostream>

void data_skimming() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("/home/matilde/Downloads/1C037D8D-2092-2448-81A4-BE32B05BFB45.root");
   
    /**
     * @brief Reads all branches names in the 'Event' TTree.
     * If it contins the word 'MET' (Missing Transverse Energy)
     * sets the branch status to 1, otherwise to 0.
     */
    TObjArray *branches = chain->GetListOfBranches();
    for(int i=0; i<branches->GetEntries(); i++) {
        TBranch *branch = (TBranch*)branches->At(i);

        TString branchName = branch->GetName();

        if (branchName.Contains("MET") && !branchName.Contains("HLT")) {
            chain->SetBranchStatus(branchName, 1);
        } else {
            chain->SetBranchStatus(branchName, 0);
        }
    }

    /**
     * @brief Selects entries that identify the run.
     * The entries have name and type of the original ones.
     */
    UInt_t run;
    UInt_t luminosityBlock;
    ULong64_t event;

    /*
     * @brief Selects the previous branches, setting their
     * status to one.
     */
    chain->SetBranchStatus("run", 1);
    chain->SetBranchStatus("luminosityBlock", 1);
    chain->SetBranchStatus("event", 1);
    
    /**
     * @brief Gets the address of the selected branches to copy
     * their values inside the new file.
     */
    chain->SetBranchAddress("run", &run);
    chain->SetBranchAddress("luminosityBlock", &luminosityBlock);
    chain->SetBranchAddress("event", &event);

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     */
    auto skimfile = std::make_unique<TFile>("skimmed.root", "RECREATE");

    /**
     * @brief Clone full TTree structure and content,
     * considering that we set some branches status to zero.
     */
    TTree *newtree = chain->CloneTree();

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}