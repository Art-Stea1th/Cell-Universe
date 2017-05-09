using System;
using System.Windows.Media;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    internal interface IEPSGenerator {

        DoubleCollection EPSCollection { get; }

        double EPS { get; set; }

        event Action NextFrameTime;
        void Start();
        void Stop();
    }
}