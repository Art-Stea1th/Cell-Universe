using System.Collections.Generic;

namespace ASD.CellUniverse.ViewModels {

    using Infrastructure.Algorithms;
    using Infrastructure.Interfaces;
    using Infrastructure.MVVM;
    using Infrastructure.SeedGenerators;
    using Infrastructure.Services;

    public sealed class ShellViewModel : BindableBase {

        private List<IMutationAlgorithm> matrixMutators;
        private int selectedMutatorIndex;

        private List<ISeedGenerator> seedWriters;
        private int selectedWriterIndex;

        private IApplicationFacade facade;



        public string Title => AppInfo.ToString();

        public IEnumerable<IMutationAlgorithm> Mutators => matrixMutators;
        public IEnumerable<ISeedGenerator> Writers => seedWriters;

        public IApplicationFacade Facade => facade;

        public int SelectedMutatorIndex {
            get => selectedMutatorIndex;
            set {
                Set(ref selectedMutatorIndex, value);
                Facade.Algorithm = matrixMutators[selectedMutatorIndex];
            }
        }

        public int SelectedWriterIndex {
            get => selectedWriterIndex;
            set {
                Set(ref selectedWriterIndex, value);
                Facade.SeedWriter = seedWriters[selectedWriterIndex];
            }
        }

        public ShellViewModel() {
            matrixMutators = new List<IMutationAlgorithm> { new TheGameOfLife(), new RandomMixer() };
            seedWriters = new List<ISeedGenerator> { new UniformRandom() };
            facade = new ApplicationFacade();
            SelectedMutatorIndex = 0;
            SelectedWriterIndex = 0;
        }

        // --- Resolutions 8x5 [16x10] ---

        //  int width = 8,   height = 5;

        //  int width = 16,  height = 10;
        //  int width = 24,  height = 15;
        //  int width = 32,  height = 20;
        //  int width = 40,  height = 25;
        //  int width = 48,  height = 30;
        //  int width = 56,  height = 35;
        //  int width = 64,  height = 40;
        //  int width = 72,  height = 45;

        //  int width = 80,  height = 50;

        //  int width = 160, height = 100;
        //  int width = 240, height = 150;
        //  int width = 320, height = 200;
        //  int width = 400, height = 250;
        //  int width = 480, height = 300;
        //  int width = 560, height = 350;
        //  int width = 640, height = 400;
        //  int width = 720, height = 450;

        //  int width = 800, height = 500;
    }
}