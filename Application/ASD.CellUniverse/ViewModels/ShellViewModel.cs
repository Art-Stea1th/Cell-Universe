using System.Collections.Generic;

namespace ASD.CellUniverse.ViewModels {

    using Infrastructure.Algorithms;
    using Infrastructure.Interfaces;
    using Infrastructure.MVVM;
    using Infrastructure.SeedGenerators;
    using Infrastructure.Services;

    public sealed class ShellViewModel : BindableBase {

        private List<IEvolutionAlgorithm> evolutionAlgorithms;
        private int selectedEvolutionIndex;

        private List<ISeedGenerator> seedWriters;
        private int selectedWriterIndex;

        private IApplicationFacade facade;

        public IEnumerable<IEvolutionAlgorithm> EvolutionAlgorithms => evolutionAlgorithms;
        public IEnumerable<ISeedGenerator> Writers => seedWriters;

        public IApplicationFacade Facade => facade;

        public int SelectedAlgorithmIndex {
            get => selectedEvolutionIndex;
            set {
                Set(ref selectedEvolutionIndex, value);
                Facade.Algorithm = evolutionAlgorithms[selectedEvolutionIndex];
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
            evolutionAlgorithms = new List<IEvolutionAlgorithm> { new Conway(), new Fredkin() };
            seedWriters = new List<ISeedGenerator> { new UniformRandom() };
            facade = new ApplicationFacade();
            SelectedAlgorithmIndex = 0;
            SelectedWriterIndex = 0;
        }        
    }
}