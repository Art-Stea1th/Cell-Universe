using System.Collections.Generic;
using System.Windows.Media;

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

        private List<BitmapScalingMode> scalingModes;
        private int selectedScalingIndex;

        private IApplicationFacade facade;

        public IEnumerable<IEvolutionAlgorithm> EvolutionAlgorithms => evolutionAlgorithms;
        public IEnumerable<ISeedGenerator> Writers => seedWriters;
        public IEnumerable<BitmapScalingMode> ScalingModes => scalingModes;
        public BitmapScalingMode SelectedScalingMode => scalingModes[selectedScalingIndex];

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

        public int SelectedScalingIndex {
            get => selectedScalingIndex;
            set {
                Set(ref selectedScalingIndex, value);
                RaisePropertyChanged(nameof(SelectedScalingMode));
            }
        }


        public ShellViewModel() {
            evolutionAlgorithms = new List<IEvolutionAlgorithm> { new Conway(), new Fredkin() };
            seedWriters = new List<ISeedGenerator> { new UniformRandom() };
            scalingModes = new List<BitmapScalingMode> { BitmapScalingMode.HighQuality, BitmapScalingMode.LowQuality, BitmapScalingMode.NearestNeighbor };
            facade = new ApplicationFacade();
            SelectedAlgorithmIndex = 0;
            SelectedWriterIndex = 0;
            SelectedScalingIndex = 0;
        }        
    }
}