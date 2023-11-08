#include <igl/opengl/glfw/Viewer.h>
#include "MeshConnectivity.h"
#include "ElasticShell.h"
#include "StaticSolve.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui.h>
#include "MidedgeAngleTanFormulation.h"
#include "MidedgeAngleSinFormulation.h"
#include "MidedgeAverageFormulation.h"
#include "StVKMaterial.h"
#include "TensionFieldStVKMaterial.h"
#include "NeoHookeanMaterial.h"
#include "RestState.h"

int numSteps;
double thickness;
double poisson;
int matid;
int sffid;

Eigen::MatrixXd curPos;
LibShell::MeshConnectivity mesh;

void repaint(igl::opengl::glfw::Viewer &viewer)
{
    viewer.data().clear();
    viewer.data().set_mesh(curPos, mesh.faces());    
}

void lameParameters(double &alpha, double &beta)
{
    double young = 1.0; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

template <class SFF>
void runSimulation(const LibShell::MeshConnectivity &mesh, 
    Eigen::MatrixXd &curPos, 
    double thickness,
    double lameAlpha,
    double lameBeta,
    int matid)
{
    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    SFF::initializeExtraDOFs(edgeDOFs, mesh, curPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);

    // initialize first fundamental forms to those of input mesh
    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, curPos, restState.abars);

    // initialize second fundamental forms to rest flat    
    restState.bbars.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++)
        restState.bbars[i].setZero();

    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);

    LibShell::MaterialModel<SFF> *mat;
    switch (matid)
    {
    case 0:
        mat = new LibShell::NeoHookeanMaterial<SFF>();
        break;
    case 1:
        mat = new LibShell::StVKMaterial<SFF>();
        break;
    case 2:
        mat = new LibShell::TensionFieldStVKMaterial<SFF>();
        break;
    default:
        assert(false);
    }

    double reg = 1e-6;
    for (int j = 1; j <= numSteps; j++)
    {
        takeOneStep(mesh, curPos, edgeDOFs, *mat, restState, reg);
    }

    delete mat;
}

int main(int argc, char *argv[])
{    
    numSteps = 30;

    // set up material parameters
    thickness = 1e-1;    
    poisson = 1.0 / 2.0;
    matid = 0;
    sffid = 0;

    // load mesh
    
    Eigen::MatrixXd origV;
    Eigen::MatrixXi F;

    std::vector<std::string> prefixes = { "./", "./example/", "../", "../example/" };

    bool found = false;
    for (auto& it : prefixes)
    {
        std::string fname = it + std::string("bunny.obj");
        if (igl::readOBJ(fname, origV, F))
        {
            found = true;
            break;            
        }
    }
    if (!found)
    {
        std::cerr << "Could not read example bunny.obj file" << std::endl;
        return -1;
    }
     
    // set up mesh connectivity
    mesh = LibShell::MeshConnectivity(F);

    // initial position
    curPos = origV;

    double lameAlpha, lameBeta;
    lameParameters(lameAlpha, lameBeta);

    switch (sffid)
    {
    case 0:
        runSimulation<LibShell::MidedgeAngleTanFormulation>(mesh, curPos, thickness, lameAlpha, lameBeta, matid);
        break;
    case 1:
        runSimulation<LibShell::MidedgeAngleSinFormulation>(mesh, curPos, thickness, lameAlpha, lameBeta, matid);
        break;
    case 2:
        runSimulation<LibShell::MidedgeAverageFormulation>(mesh, curPos, thickness, lameAlpha, lameBeta, matid);
        break;
    default:
        assert(false);
    }
}
