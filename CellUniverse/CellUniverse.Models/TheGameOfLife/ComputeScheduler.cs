using System;
using System.Threading;
using System.Collections.Generic;
using System.Collections.Concurrent;

namespace CellUniverse.Models.TheGameOfLife {

    internal sealed class ComputeScheduler : IDisposable {

        private Cell[] universePlacement;

        private int precalculatedBufferCount;
        private int precalculatedBufferLimit;

        private ConcurrentQueue<ConcurrentQueue<Tuple<int, int, bool>>> precalculatedBuffer;

        private bool thisWillBeDestroyed;

        internal ComputeScheduler(Cell[] universe, int bufferSize = 3) {
            universePlacement = universe;
            precalculatedBufferLimit = bufferSize;
            precalculatedBufferCount = 0;
            precalculatedBuffer = new ConcurrentQueue<ConcurrentQueue<Tuple<int, int, bool>>>();
            thisWillBeDestroyed = false;
        }

        public void Dispose() {
            thisWillBeDestroyed = true;
        }

        internal void Start() {

            ThreadPool.QueueUserWorkItem((object o) => {

                while (!thisWillBeDestroyed) {
                    if (precalculatedBufferCount < precalculatedBufferLimit) {

                        var localBuffer = new ConcurrentQueue<Tuple<int, int, bool>>();

                        for (int i = 0; i < universePlacement.Length; ++i) {
                            Cell nextCell = universePlacement[i];
                            if (nextCell.CalculateNextState()) {
                                localBuffer.Enqueue(new Tuple<int, int, bool>(nextCell.X, nextCell.Y, nextCell));
                            }
                        }
                        for (int i = 0; i < universePlacement.Length; i++) {
                            universePlacement[i].ApplyNextState();
                        }

                        precalculatedBuffer.Enqueue(localBuffer);
                        Interlocked.Increment(ref precalculatedBufferCount);
                    }
                    else {
                        Thread.Sleep(1);
                    }
                }

            }, null);            
        }

        internal IEnumerable<Tuple<int, int, bool>> NextGeneration {
            get {
                ConcurrentQueue<Tuple<int, int, bool>> nextGeneration = null;
                bool success = false;

                while (!success) {

                    if (precalculatedBuffer.TryDequeue(out nextGeneration)) {

                        while (nextGeneration.Count > 0) {
                            Tuple<int, int, bool> next = null;
                            if (nextGeneration.TryDequeue(out next)) {
                                yield return next;
                            }
                        }
                        success = true;
                        Interlocked.Decrement(ref precalculatedBufferCount);
                    }
                    else {
                        Thread.Sleep(1);
                    }
                }
            }
        }
    }
}