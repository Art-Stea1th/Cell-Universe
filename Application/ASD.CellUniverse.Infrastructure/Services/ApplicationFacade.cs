using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Services {

    using Controllers;
    using Interfaces;
    using MVVM;

    public class ApplicationFacade : BindableBase, IApplicationFacade {

        private uint[,] matrix;
        private int generationWidth;
        private int generationHeight;
        private bool matrixReadyToChange;
        private bool matrixReadyToMutate;

        private IGenerationController controller;
        private IEPSGenerator epsGenerator;

        private ISeedGenerator seedWriter;

        private IEvolutionAlgorithm algorithm;

        public uint[,] Matrix {
            get => matrix;
            private set => Set(ref matrix, value);
        }

        public int GenerationWidth {
            get => generationWidth;
            set {
                if (matrixReadyToChange) {
                    Set(ref generationWidth, ValidWidth(value));
                    ChangeResolution(generationWidth, generationHeight);
                }
            }
        }

        public int GenerationHeight {
            get => generationHeight;
            set {
                if (matrixReadyToChange) {
                    Set(ref generationHeight, ValidHeight(value));
                    ChangeResolution(generationWidth, generationHeight);
                }
            }
        }

        public bool MatrixReadyToChange {
            get => matrixReadyToChange;
            private set => Set(ref matrixReadyToChange, value);
        }

        public bool MatrixReadyToMutate {
            get => matrixReadyToMutate;
            private set => Set(ref matrixReadyToMutate, value);
        }

        public ISeedGenerator SeedWriter {
            get => seedWriter;
            set => Set(ref seedWriter, value);
        }

        public ICommand WriteSeed => new RelayCommand(
            (o) => {
                Matrix = SeedWriter.GenerateNew(matrix.GetLength(0), matrix.GetLength(1));
                MatrixReadyToMutate = true;
            },
            (o) => MatrixReadyToChange);

        public IEvolutionAlgorithm Algorithm {
            get => algorithm;
            set => Set(ref algorithm, value);
        }

        public DoubleCollection EPSCollection => epsGenerator.EPSCollection;
        public double MinEPS => epsGenerator.EPSCollection.First();
        public double MaxEPS => epsGenerator.EPSCollection.Last();
        public double EPS { get => epsGenerator.EPS; set => epsGenerator.EPS = value; }

        public State State => controller.State;
        public ICommand Start => controller.Start;
        public ICommand Stop => controller.Stop;

        public ApplicationFacade() {
            Initialize(new EPSGenerationService(), new GenerationStateMachine());
            MatrixReadyToChange = State == State.Stopped;
            GenerationWidth = 320;
            GenerationHeight = 180;
        }

        private void Initialize(IEPSGenerator epsGenerator, IGenerationController controller) {
            ConfigureFPSGenerator(this.epsGenerator = epsGenerator);
            ConfigureController(this.controller = controller);
        }

        private void ConfigureFPSGenerator(IEPSGenerator epsGenerator) {
            epsGenerator.NextFrameTime += () => Matrix = algorithm.Mutate(matrix);
        }

        private void ConfigureController(IGenerationController controller) {
            controller.PropertyChanged += (s, e) => {
                RaisePropertyChanged(e.PropertyName);
                MatrixReadyToChange = State == State.Stopped;
            };
            controller.Started += epsGenerator.Start;
            controller.Paused += epsGenerator.Stop;
            controller.Resumed += epsGenerator.Start;
            controller.Stopped += async () => await OnStop(); // tmp
            controller.Reseted += async () => await OnStop(); // tmp
        }

        private Task OnStop() { // tmp
            epsGenerator.Stop();
            Matrix = null;
            var resultTask = new Task(() => {
                ChangeResolution(GenerationWidth, GenerationHeight);
            });
            resultTask.Start();
            return resultTask;
        }

        private void ChangeResolution(int width, int height) {
            Matrix = new uint[width, height];
            MatrixReadyToMutate = false;
        }

        private int ValidWidth(int value) => Valid(value, 1, 1920);
        private int ValidHeight(int value) => Valid(value, 1, 1080);
        private int Valid(int value, int min, int max) => value < min ? min : value > max ? max : value;
    }
}