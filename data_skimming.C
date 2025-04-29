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
    //chain->Add("datasets/2E96A5E9-C938-A149-BBBF-8FD81A9E5AD6.root"); // 0
    chain->Add("datasets/6357E7BC-502C-2E45-A649-73A57B651715.root");  // 1
   
    /**
     * @brief Reads all branches names in the 'Event' TTree.
     * Cleans data by excluding trigger entries.
     */
    TObjArray *branches = chain->GetListOfBranches();
    for(int i=0; i<branches->GetEntries(); i++) {
        TBranch *branch = (TBranch*)branches->At(i);

        TString branchName = branch->GetName();

        if (branchName.Contains("MET") &&
            !branchName.BeginsWith("Calo") &&
            !branchName.BeginsWith("Chs") &&
            !branchName.Contains("CorrT1") &&
            !branchName.BeginsWith("Deep") &&
            !branchName.Contains("Unclust") &&
            !branchName.Contains("Puppi") &&
            !branchName.BeginsWith("Raw") &&
            !branchName.BeginsWith("Tk") &&
            !branchName.Contains("fiducial") &&
            !branchName.BeginsWith("Flag") &&
            !branchName.Contains("HLT")
        ) {
            chain->SetBranchStatus(branchName, 1);
        }
        else if (
            branchName.Contains("Jet") &&
            !branchName.Contains("btag") &&
            !branchName.Contains("ch") &&
            !branchName.Contains("hfsigma") &&
            !branchName.Contains("Reg") &&
            !branchName.Contains("electron") &&
            !branchName.Contains("hf") &&
            !branchName.Contains("muon") &&
            !branchName.Contains("Constituents") &&
            !branchName.Contains("SoftActivity") &&
            !branchName.Contains("CorrT1") &&
            !branchName.Contains("btag") &&
            !branchName.Contains("EF") &&
            !branchName.Contains("Flavour") &&
            !branchName.Contains("cleanmask") &&
            !branchName.Contains("Id") &&
            !branchName.Contains("L1") &&
            !branchName.Contains("HLT")
        ) {
            chain->SetBranchStatus(branchName, 1);
        } else {
            chain->SetBranchStatus(branchName, 0);
        }
    }

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("datasets/skimmed1.root", "RECREATE");

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