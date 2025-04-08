#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void data_skimming() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("/home/matilde/Downloads/1C037D8D-2092-2448-81A4-BE32B05BFB45.root");

    /**
     * @brief Selects entries that identify the run.
     * The entries have name and type of the original ones.
     */
    UInt_t run;
    UInt_t luminosityBlock;
    ULong64_t event;

    /**
     * @brief Sets all branch statuses at zero.
     */
    chain->SetBranchStatus("*", 0);

    /**
     * @brief Selects only interesting branches setting their
     * status at one.
     */
    chain->SetBranchStatus("run", 1);
    chain->SetBranchStatus("luminosityBlock", 1);
    chain->SetBranchStatus("event", 1);
    
    /**
     * @brief Gets the address of the selected branches to copy
     * their valuesinside the new file.
     */
    chain->SetBranchAddress("run", &run);
    chain->SetBranchAddress("luminosityBlock", &luminosityBlock);
    chain->SetBranchAddress("event", &event);

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     */
    auto skimfile = std::make_unique<TFile>("skimmed.root", "RECREATE");

    //chain->LoadTree(0); only usefull in a for loop
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