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
    }
}