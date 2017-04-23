using System;
using System.ComponentModel;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    interface IFrameSequenceGenerator : INotifyPropertyChanged {

        event Action<bool[,]> NextFrameReady;
        double FPS { get; set; }

        void Start();
        void Pause();
        void Stop();
    }
}