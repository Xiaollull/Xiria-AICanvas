import { useId } from "react";

export function BrandLogo({ className = "" }) {
  const prefix = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const notchId = `${prefix}-notch`;
  const maskId = `${prefix}-mask`;
  const foldId = `${prefix}-fold`;
  const tickId = `${prefix}-tick`;
  return <svg className={`brand-logo ${className}`.trim()} viewBox="0 0 1024 1024" aria-hidden="true" focusable="false">
    <defs>
      <path id={notchId} d="M 478 40 H 546 V 94 L 512 122 L 478 94 Z" />
      <mask id={maskId}><rect width="1024" height="1024" fill="#000" /><circle cx="512" cy="512" r="456" fill="#fff" /><g fill="#000"><use href={`#${notchId}`} /><use href={`#${notchId}`} transform="rotate(90 512 512)" /><use href={`#${notchId}`} transform="rotate(180 512 512)" /><use href={`#${notchId}`} transform="rotate(270 512 512)" /></g></mask>
      <path id={foldId} d="M 342 286 H 444 L 494 336 V 410 L 464 380 H 398 V 438 L 424 464 L 394 494 H 336 L 286 444 V 342 Z" />
      <path id={tickId} d="M 207 207 L 223 223" />
    </defs>
    <circle cx="512" cy="512" r="456" fill="var(--logo-accent)" mask={`url(#${maskId})`} />
    <circle cx="512" cy="512" r="434" fill="var(--logo-disc)" mask={`url(#${maskId})`} />
    <circle cx="512" cy="512" r="414" fill="none" stroke="var(--logo-guide)" strokeWidth="3" />
    <g fill="none" stroke="var(--logo-ink)" strokeWidth="5"><path d="M 524 132 A 380 380 0 0 1 892 500" /><path d="M 892 524 A 380 380 0 0 1 524 892" /><path d="M 500 892 A 380 380 0 0 1 132 524" /><path d="M 132 500 A 380 380 0 0 1 500 132" /></g>
    <g fill="none" stroke="var(--logo-ink)" strokeWidth="9"><use href={`#${tickId}`} /><use href={`#${tickId}`} transform="rotate(90 512 512)" /><use href={`#${tickId}`} transform="rotate(180 512 512)" /><use href={`#${tickId}`} transform="rotate(270 512 512)" /></g>
    <g fill="none" stroke="var(--logo-accent)" strokeWidth="5"><path d="M 395 232 A 303 303 0 0 1 629 232" /><path d="M 792 395 A 303 303 0 0 1 792 629" /><path d="M 629 792 A 303 303 0 0 1 395 792" /><path d="M 232 629 A 303 303 0 0 1 232 395" /></g>
    <use href={`#${foldId}`} fill="var(--logo-accent)" stroke="var(--logo-ink)" strokeWidth="5" strokeLinejoin="miter" />
    <use href={`#${foldId}`} fill="var(--logo-ink)" stroke="var(--logo-accent)" strokeWidth="5" strokeLinejoin="miter" transform="rotate(90 512 512)" />
    <use href={`#${foldId}`} fill="var(--logo-accent)" stroke="var(--logo-ink)" strokeWidth="5" strokeLinejoin="miter" transform="rotate(180 512 512)" />
    <use href={`#${foldId}`} fill="var(--logo-ink)" stroke="var(--logo-accent)" strokeWidth="5" strokeLinejoin="miter" transform="rotate(270 512 512)" />
    <path fill="var(--logo-ink)" stroke="var(--logo-accent)" strokeWidth="5" strokeLinejoin="miter" d="M 432 432 H 461 L 512 483 L 563 432 H 592 V 461 L 541 512 L 592 563 V 592 H 563 L 512 541 L 461 592 H 432 V 563 L 483 512 L 432 461 Z" />
  </svg>;
}
export function LoadingLogo() {
  return <svg className="loader-logo" viewBox="0 0 1024 1024" aria-hidden="true" focusable="false">
    <defs>
      <path id="loader-backplate-notch" d="M 478 40 H 546 V 94 L 512 122 L 478 94 Z" />
      <mask id="loader-backplate-mask">
        <rect width="1024" height="1024" fill="#000" />
        <circle cx="512" cy="512" r="456" fill="#fff" />
        <g fill="#000">
          <use href="#loader-backplate-notch" />
          <use href="#loader-backplate-notch" transform="rotate(90 512 512)" />
          <use href="#loader-backplate-notch" transform="rotate(180 512 512)" />
          <use href="#loader-backplate-notch" transform="rotate(270 512 512)" />
        </g>
      </mask>
      <path id="loader-fold" d="M 342 286 H 444 L 494 336 V 410 L 464 380 H 398 V 438 L 424 464 L 394 494 H 336 L 286 444 V 342 Z" />
      <path id="loader-orbit-tick" d="M 207 207 L 223 223" />
    </defs>

    <g fill="none">
      <circle className="loader-orbit loader-orbit-a" cx="512" cy="512" r="472" stroke="var(--logo-ink)" strokeWidth="5" strokeDasharray="1720 1260" />
      <circle className="loader-orbit loader-orbit-b" cx="512" cy="512" r="487" stroke="var(--logo-accent)" strokeWidth="4" strokeDasharray="1286 1790" />
      <circle className="loader-orbit loader-orbit-c" cx="512" cy="512" r="501" stroke="var(--logo-guide)" strokeWidth="3" strokeDasharray="818 2345" />
    </g>

    <circle cx="512" cy="512" r="456" fill="var(--logo-accent)" mask="url(#loader-backplate-mask)" />
    <circle cx="512" cy="512" r="434" fill="var(--logo-disc)" mask="url(#loader-backplate-mask)" />
    <circle cx="512" cy="512" r="414" fill="none" stroke="var(--logo-guide)" strokeWidth="3" />

    <g fill="none" stroke="var(--logo-ink)" strokeWidth="5">
      <path d="M 524 132 A 380 380 0 0 1 892 500" />
      <path d="M 892 524 A 380 380 0 0 1 524 892" />
      <path d="M 500 892 A 380 380 0 0 1 132 524" />
      <path d="M 132 500 A 380 380 0 0 1 500 132" />
    </g>

    <g fill="none" stroke="var(--logo-ink)" strokeWidth="9">
      <use href="#loader-orbit-tick" />
      <use href="#loader-orbit-tick" transform="rotate(90 512 512)" />
      <use href="#loader-orbit-tick" transform="rotate(180 512 512)" />
      <use href="#loader-orbit-tick" transform="rotate(270 512 512)" />
    </g>

    <g className="loader-motion-arcs" fill="none" stroke="var(--logo-accent)" strokeWidth="5">
      <path d="M 395 232 A 303 303 0 0 1 629 232" />
      <path d="M 792 395 A 303 303 0 0 1 792 629" />
      <path d="M 629 792 A 303 303 0 0 1 395 792" />
      <path d="M 232 629 A 303 303 0 0 1 232 395" />
    </g>

    <use href="#loader-fold" fill="var(--logo-accent)" stroke="var(--logo-ink)" strokeWidth="5" strokeLinejoin="miter" />
    <use href="#loader-fold" fill="var(--logo-ink)" stroke="var(--logo-accent)" strokeWidth="5" strokeLinejoin="miter" transform="rotate(90 512 512)" />
    <use href="#loader-fold" fill="var(--logo-accent)" stroke="var(--logo-ink)" strokeWidth="5" strokeLinejoin="miter" transform="rotate(180 512 512)" />
    <use href="#loader-fold" fill="var(--logo-ink)" stroke="var(--logo-accent)" strokeWidth="5" strokeLinejoin="miter" transform="rotate(270 512 512)" />
    <path fill="var(--logo-ink)" stroke="var(--logo-accent)" strokeWidth="5" strokeLinejoin="miter" d="M 432 432 H 461 L 512 483 L 563 432 H 592 V 461 L 541 512 L 592 563 V 592 H 563 L 512 541 L 461 592 H 432 V 563 L 483 512 L 432 461 Z" />
  </svg>;
}
