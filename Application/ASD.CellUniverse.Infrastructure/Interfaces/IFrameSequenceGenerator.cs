using System;
using System.ComponentModel;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IFrameSequenceGenerator : INotifyPropertyChanged {

        event Action<bool[,]> NextFrameReady;
        double FPS { get; set; }

        IGenerationAlgorithm GenerationAlgorithm { get; set; }

        bool[,] GeneratedData { get; set; }

        void Start();
        void Pause();
        void Stop();
    }
}