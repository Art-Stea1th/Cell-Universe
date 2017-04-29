using System;
using System.Collections.Generic;

namespace ASD.CellUniverse.ViewModels {

    using Infrastructure.Algorithms;
    using Infrastructure.Controllers;
    using Infrastructure.Interfaces;
    using Infrastructure.MVVM;
    using Infrastructure.Services;

    public sealed class ShellViewModel : BindableBase {

        public string Title => AppInfo.ToString();

        private int mutatorSelectedIndex = 0;
        public int MutatorSelectedIndex {
            get => mutatorSelectedIndex;
            set {
                SetProperty(ref mutatorSelectedIndex, value);
                SequenceGenerator.GenerationAlgorithm = matrixMutators[mutatorSelectedIndex];
            }
        }
        public IEnumerable<IMatrixMutator> Mutators => matrixMutators;
        private List<IMatrixMutator> matrixMutators;

        public IFrameSequenceGenerator SequenceGenerator { get; private set; }

        public IMainController Controller { get; private set; }

        private bool canResolutionChange;

        public bool CanResolutionChange {
            get => canResolutionChange;
            set => SetProperty(ref canResolutionChange, value);
        }

        // --- TEMP >> ---



        //int width = 800, height = 500; // 16 : 10
        //int width = 480, height = 300; // 16 : 10
        //int width = 400, height = 250; // 16 : 10
        int width = 320, height = 200; // 16 : 10
        //int width = 160, height = 100; // 16 : 10
        //int width = 80, height = 50; // 16 : 10
        //int width = 40, height = 25; // 16 : 10
        //int width = 32, height = 20; // 16 : 10
        //int width = 16, height = 10; // 16 : 10
        //int width = 8, height = 5; // 16 : 10


        private byte[,] intencityData;
        public byte[,] IntencityData {
            get => intencityData;
            set => SetProperty(ref intencityData, value);
        }

        // --- << TEMP ---

        public ShellViewModel() {

            IntencityData = new byte[width, height];

            matrixMutators = new List<IMatrixMutator> { new TheGameOfLife(), new RandomMixer() };

            SequenceGenerator = new FrameGenerationService(matrixMutators[mutatorSelectedIndex]);

            SequenceGenerator.NextFrameReady += (a) => UpdateIntencityData(a);
            SequenceGenerator.GeneratedData = CreateRandom(width, height);

            Controller = new ApplicationStateMachine();
            Controller.Started += SequenceGenerator.Play;
            Controller.Paused += SequenceGenerator.Pause;
            Controller.Resumed += SequenceGenerator.Resume;
            Controller.Stopped += SequenceGenerator.Stop;
            Controller.Reseted += SequenceGenerator.Reset;
            Controller.StateChanged += (s) => CanResolutionChange = s == State.Stopped ? true : false;

            CanResolutionChange = Controller.State == State.Stopped;

            // TMP
            Controller.Stopped += () => SequenceGenerator.GeneratedData = CreateRandom(width, height);
            Controller.Reseted += () => SequenceGenerator.GeneratedData = CreateRandom(width, height);

        }

        private void UpdateIntencityData(byte[,] newIntencityData)
            => IntencityData = newIntencityData;

        // TMP
        private byte[,] CreateRandom(int width, int height) {

            var random = new Random();
            var result = new byte[width, height];

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    result[x, y] = random.Next() % 2 == 1 ? (byte)255 : (byte)0;
                }
            }
            return result;
        }
    }
}