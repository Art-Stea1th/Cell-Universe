using System.Collections.Generic;
using System.Windows.Media;

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

        private List<BitmapScalingMode> scalingModes;
        private int selectedScalingIndex;

        private IApplicationFacade facade;

        public IEnumerable<IMutationAlgorithm> Mutators => matrixMutators;
        public IEnumerable<ISeedGenerator> Writers => seedWriters;
        public IEnumerable<BitmapScalingMode> ScalingModes => scalingModes;
        public BitmapScalingMode SelectedScalingMode => scalingModes[selectedScalingIndex];

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

        public int SelectedScalingIndex {
            get => selectedScalingIndex;
            set {
                Set(ref selectedScalingIndex, value);
                RaisePropertyChanged(nameof(SelectedScalingMode));
            }
        }


        public ShellViewModel() {
            matrixMutators = new List<IMutationAlgorithm> { new TheGameOfLife(), new RandomMixer() };
            seedWriters = new List<ISeedGenerator> { new UniformRandom() };
            scalingModes = new List<BitmapScalingMode> { BitmapScalingMode.HighQuality, BitmapScalingMode.LowQuality, BitmapScalingMode.NearestNeighbor };
            facade = new ApplicationFacade();
            SelectedMutatorIndex = 0;
            SelectedWriterIndex = 0;
            SelectedScalingIndex = 0;
        }        
    }
}